"""
GPU-accelerated Roomba physics simulation using PyTorch.

Simulates multiple Roombas with rigid-body 2D physics:
- Continuous position, velocity, and angular dynamics
- Wall and obstacle collisions with restitution
- Friction (linear and angular drag)
- Coverage heatmap tracked on GPU
- Batch-parallel: simulate hundreds of Roombas simultaneously

Runs on CUDA if available, otherwise falls back to CPU.
"""

import torch
import torch.nn.functional as F
import math
import argparse
import time


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PhysicsRoom:
    """
    A 2D room stored as a GPU tensor occupancy grid.
    0 = free space, 1 = obstacle/wall.
    Coverage is tracked as a float heatmap.
    """

    def __init__(self, width, height, resolution, obstacle_fraction, device):
        self.width = width      # meters
        self.height = height    # meters
        self.res = resolution   # cells per meter
        self.device = device

        self.grid_w = int(width * resolution)
        self.grid_h = int(height * resolution)

        # Occupancy: 1=obstacle, 0=free
        self.occupancy = torch.zeros(self.grid_h, self.grid_w, device=device)
        self._add_walls()
        self._add_obstacles(obstacle_fraction)

        # Coverage heatmap (how many times each cell visited)
        self.coverage = torch.zeros(self.grid_h, self.grid_w, device=device)

        # Precompute free-cell count for coverage stats
        self.free_cells = (self.occupancy == 0).sum().float()

    def _add_walls(self):
        self.occupancy[0, :] = 1
        self.occupancy[-1, :] = 1
        self.occupancy[:, 0] = 1
        self.occupancy[:, -1] = 1

    def _add_obstacles(self, fraction):
        interior = self.occupancy[1:-1, 1:-1]
        num_obs = int(interior.numel() * fraction)
        indices = torch.randperm(interior.numel(), device=self.device)[:num_obs]
        interior_flat = interior.reshape(-1)
        interior_flat[indices] = 1

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        gx = (x * self.res).long().clamp(0, self.grid_w - 1)
        gy = (y * self.res).long().clamp(0, self.grid_h - 1)
        return gx, gy

    def is_obstacle_at(self, x, y):
        """Check if world positions collide with obstacles. Batched."""
        gx, gy = self.world_to_grid(x, y)
        return self.occupancy[gy, gx] > 0.5

    def mark_covered(self, x, y, radius_cells=2):
        """Mark cells around each Roomba position as covered."""
        gx, gy = self.world_to_grid(x, y)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    cx = (gx + dx).clamp(0, self.grid_w - 1)
                    cy = (gy + dy).clamp(0, self.grid_h - 1)
                    self.coverage[cy, cx] = 1.0

    def get_coverage_pct(self):
        covered = ((self.coverage > 0) & (self.occupancy == 0)).sum().float()
        return (covered / self.free_cells * 100).item()


class RoombaBatch:
    """
    Batch of N Roombas simulated in parallel on GPU with rigid-body physics.

    Each Roomba has:
      - position (x, y) in meters
      - velocity (vx, vy) in m/s
      - heading angle (theta) in radians
      - angular velocity (omega) in rad/s
    """

    def __init__(self, n, room, radius=0.17, max_speed=0.33, device=None):
        self.n = n
        self.room = room
        self.device = device or room.device
        self.radius = radius        # Roomba radius in meters
        self.max_speed = max_speed  # ~0.33 m/s for a real Roomba

        # Physics constants
        self.linear_drag = 0.5      # velocity damping
        self.angular_drag = 2.0     # angular damping
        self.drive_force = 1.5      # forward drive acceleration (m/s^2)
        self.restitution = 0.3      # bounce coefficient
        self.turn_torque = 5.0      # turning acceleration (rad/s^2)

        # State tensors [N]
        self.x = torch.empty(n, device=self.device)
        self.y = torch.empty(n, device=self.device)
        self.vx = torch.zeros(n, device=self.device)
        self.vy = torch.zeros(n, device=self.device)
        self.theta = torch.empty(n, device=self.device).uniform_(0, 2 * math.pi)
        self.omega = torch.zeros(n, device=self.device)

        # Behavioral state: time since last collision (triggers random turn)
        self.turn_timer = torch.zeros(n, device=self.device)
        self.turn_direction = torch.ones(n, device=self.device)  # +1 or -1

        self._place_initial_positions()

    def _place_initial_positions(self):
        """Place Roombas at random free positions."""
        for i in range(self.n):
            while True:
                px = torch.empty(1, device=self.device).uniform_(
                    self.radius + 0.1, self.room.width - self.radius - 0.1)
                py = torch.empty(1, device=self.device).uniform_(
                    self.radius + 0.1, self.room.height - self.radius - 0.1)
                if not self.room.is_obstacle_at(px, py).item():
                    self.x[i] = px
                    self.y[i] = py
                    break

    def step(self, dt):
        """Advance physics by dt seconds. Fully batched on GPU."""

        # --- Drive force: Roombas drive forward along heading ---
        fx = self.drive_force * torch.cos(self.theta)
        fy = self.drive_force * torch.sin(self.theta)

        # --- Apply turning when timer is active ---
        turning = self.turn_timer > 0
        torque = torch.where(turning,
                             self.turn_direction * self.turn_torque,
                             torch.zeros_like(self.omega))

        # Decrease turn timer
        self.turn_timer = (self.turn_timer - dt).clamp(min=0)

        # --- Integrate angular dynamics ---
        self.omega += torque * dt
        self.omega -= self.angular_drag * self.omega * dt
        self.theta += self.omega * dt

        # --- Integrate linear dynamics ---
        self.vx += fx * dt - self.linear_drag * self.vx * dt
        self.vy += fy * dt - self.linear_drag * self.vy * dt

        # Clamp speed
        speed = torch.sqrt(self.vx ** 2 + self.vy ** 2)
        scale = torch.clamp(self.max_speed / (speed + 1e-8), max=1.0)
        self.vx *= scale
        self.vy *= scale

        # --- Tentative new position ---
        new_x = self.x + self.vx * dt
        new_y = self.y + self.vy * dt

        # --- Wall/obstacle collision detection ---
        # Check multiple probe points around the Roomba circumference
        collided = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        normal_x = torch.zeros(self.n, device=self.device)
        normal_y = torch.zeros(self.n, device=self.device)

        num_probes = 8
        for k in range(num_probes):
            angle = 2 * math.pi * k / num_probes
            probe_x = new_x + self.radius * math.cos(angle)
            probe_y = new_y + self.radius * math.sin(angle)
            hit = self.room.is_obstacle_at(probe_x, probe_y)
            collided |= hit
            # Accumulate collision normal (points away from obstacle)
            normal_x -= hit.float() * math.cos(angle)
            normal_y -= hit.float() * math.sin(angle)

        # Also check room boundary collisions
        hit_left = new_x < self.radius
        hit_right = new_x > self.room.width - self.radius
        hit_bottom = new_y < self.radius
        hit_top = new_y > self.room.height - self.radius

        boundary_hit = hit_left | hit_right | hit_bottom | hit_top
        collided |= boundary_hit
        normal_x += hit_left.float() - hit_right.float()
        normal_y += hit_bottom.float() - hit_top.float()

        # Normalize collision normal
        norm_len = torch.sqrt(normal_x ** 2 + normal_y ** 2) + 1e-8
        normal_x /= norm_len
        normal_y /= norm_len

        # --- Reflect velocity on collision ---
        v_dot_n = self.vx * normal_x + self.vy * normal_y
        reflect_mask = collided & (v_dot_n < 0)
        self.vx = torch.where(reflect_mask,
                              self.vx - (1 + self.restitution) * v_dot_n * normal_x,
                              self.vx)
        self.vy = torch.where(reflect_mask,
                              self.vy - (1 + self.restitution) * v_dot_n * normal_y,
                              self.vy)

        # On collision, trigger a random turn (like a real Roomba)
        newly_collided = collided & (self.turn_timer == 0)
        if newly_collided.any():
            self.turn_timer = torch.where(
                newly_collided,
                torch.empty_like(self.turn_timer).uniform_(0.5, 2.0),
                self.turn_timer)
            self.turn_direction = torch.where(
                newly_collided,
                (torch.randint(0, 2, (self.n,), device=self.device).float() * 2 - 1),
                self.turn_direction)

        # --- Update positions (revert if collided) ---
        self.x = torch.where(collided, self.x, new_x)
        self.y = torch.where(collided, self.y, new_y)

        # Clamp to room bounds
        self.x.clamp_(self.radius, self.room.width - self.radius)
        self.y.clamp_(self.radius, self.room.height - self.radius)

        # --- Mark coverage ---
        radius_cells = max(1, int(self.radius * self.room.res))
        self.room.mark_covered(self.x, self.y, radius_cells)


def render_ascii(room, roombas, max_cols=80, max_rows=40):
    """Render a downsampled ASCII view of the room."""
    occ = room.occupancy.cpu()
    cov = room.coverage.cpu()
    h, w = occ.shape

    step_x = max(1, w // max_cols)
    step_y = max(1, h // max_rows)

    # Roomba grid positions
    gx, gy = room.world_to_grid(roombas.x, roombas.y)
    roomba_set = set()
    for i in range(roombas.n):
        rx = gx[i].item() // step_x
        ry = gy[i].item() // step_y
        roomba_set.add((rx, ry))

    lines = []
    for row in range(0, h, step_y):
        line = ''
        for col in range(0, w, step_x):
            dr = min(row + step_y, h)
            dc = min(col + step_x, w)
            cell = (col // step_x, row // step_y)

            if cell in roomba_set:
                line += 'R'
            elif occ[row:dr, col:dc].any():
                line += '#'
            elif cov[row:dr, col:dc].any():
                line += '.'
            else:
                line += ' '
        lines.append(line)
    return '\n'.join(lines)


def run_simulation(args):
    device = get_device()
    print(f"Device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    room = PhysicsRoom(
        width=args.room_width,
        height=args.room_height,
        resolution=args.resolution,
        obstacle_fraction=args.obstacles,
        device=device,
    )
    print(f"Room: {args.room_width}x{args.room_height}m, "
          f"grid: {room.grid_w}x{room.grid_h}, "
          f"obstacles: {args.obstacles*100:.0f}%")

    roombas = RoombaBatch(
        n=args.num_roombas,
        room=room,
        radius=args.roomba_radius,
        max_speed=args.max_speed,
        device=device,
    )
    print(f"Roombas: {args.num_roombas}, radius: {args.roomba_radius}m, "
          f"max speed: {args.max_speed} m/s")

    dt = args.dt
    total_steps = int(args.sim_time / dt)
    report_interval = max(1, total_steps // 10)

    print(f"Simulating {args.sim_time}s ({total_steps} steps at dt={dt}s)...\n")

    t0 = time.time()
    for step in range(total_steps):
        roombas.step(dt)

        if args.verbose and (step + 1) % report_interval == 0:
            cov = room.get_coverage_pct()
            elapsed = time.time() - t0
            sim_time = (step + 1) * dt
            print(f"  t={sim_time:6.1f}s  |  coverage: {cov:5.1f}%  |  "
                  f"wall time: {elapsed:.2f}s")

    wall_time = time.time() - t0
    final_cov = room.get_coverage_pct()

    print()
    if args.verbose:
        print(render_ascii(room, roombas))
        print()

    print(f"Final coverage: {final_cov:.1f}%")
    print(f"Wall time: {wall_time:.2f}s  "
          f"({total_steps / wall_time:.0f} steps/s)")

    return final_cov


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated Roomba physics simulation")

    # Room
    parser.add_argument("--room-width", type=float, default=10.0,
                        help="Room width in meters (default: 10)")
    parser.add_argument("--room-height", type=float, default=10.0,
                        help="Room height in meters (default: 10)")
    parser.add_argument("--resolution", type=int, default=10,
                        help="Grid cells per meter (default: 10)")
    parser.add_argument("--obstacles", type=float, default=0.05,
                        help="Obstacle fraction 0-1 (default: 0.05)")

    # Roombas
    parser.add_argument("--num-roombas", type=int, default=4,
                        help="Number of Roombas (default: 4)")
    parser.add_argument("--roomba-radius", type=float, default=0.17,
                        help="Roomba radius in meters (default: 0.17)")
    parser.add_argument("--max-speed", type=float, default=0.33,
                        help="Max Roomba speed m/s (default: 0.33)")

    # Simulation
    parser.add_argument("--sim-time", type=float, default=120.0,
                        help="Simulation duration in seconds (default: 120)")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Timestep in seconds (default: 0.05)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Show progress and final ASCII render")

    args = parser.parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
