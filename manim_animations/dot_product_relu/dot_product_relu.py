"""
3Blue1Brown-style Manim animation demonstrating:
- A random unit vector
- Dot products with 10 random unit vectors on a unit sphere
- ReLU (positive portion only) of each dot product
- Accumulation of the ReLU'd values

Animation proceeds in segments:
1. Setup: Show the 3D unit sphere and the primary vector
2. Show each of the 10 random vectors one at a time
3. Compute dot product, apply ReLU, and accumulate
4. Final summary
"""

from manim import *
import numpy as np

# Seed for reproducibility in the animation
np.random.seed(42)

# Generate random unit vectors on the unit sphere
def random_unit_vector():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)

# Pre-generate vectors
PRIMARY_VECTOR = random_unit_vector()
RANDOM_VECTORS = [random_unit_vector() for _ in range(10)]

# Color palette
PRIMARY_COLOR = YELLOW
POSITIVE_COLOR = GREEN
NEGATIVE_COLOR = RED
ACCENT_COLOR = BLUE


class DotProductReLUScene(ThreeDScene):
    """Main animation: 3D dot product with ReLU accumulation."""

    def construct(self):
        # ── Segment 1: Title ──
        self.segment_title()

        # ── Segment 2: Setup sphere and primary vector ──
        self.segment_setup()

        # ── Segment 3: Iterate through each vector, compute dot & ReLU ──
        self.segment_dot_products()

        # ── Segment 4: Final summary ──
        self.segment_summary()

    def segment_title(self):
        title = Text("Dot Product with ReLU", font_size=48)
        subtitle = Text(
            "Accumulating positive projections onto a unit vector",
            font_size=24,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.4)
        group = VGroup(title, subtitle)

        self.play(FadeIn(group, shift=UP * 0.5))
        self.wait(1.5)
        self.play(FadeOut(group))
        self.wait(0.5)

    def segment_setup(self):
        # Set up 3D camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        # Axes
        self.axes = ThreeDAxes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[-1.5, 1.5, 0.5],
            x_length=6,
            y_length=6,
            z_length=6,
        )

        # Unit sphere as wireframe great circles (lightweight)
        sphere_circles = VGroup()
        # Latitude circles
        for angle in np.linspace(-PI / 3, PI / 3, 4):
            r = np.cos(angle)
            z = np.sin(angle)
            circle = Circle(radius=r * 2, color=BLUE_E, stroke_width=0.8, stroke_opacity=0.4)
            circle.move_to(self.axes.c2p(0, 0, z))
            circle.rotate(PI / 2, axis=RIGHT, about_point=self.axes.c2p(0, 0, z))
            sphere_circles.add(circle)
        # Longitude circles
        for angle in np.linspace(0, PI, 6, endpoint=False):
            circle = Circle(radius=2, color=BLUE_E, stroke_width=0.8, stroke_opacity=0.4)
            circle.rotate(PI / 2, axis=RIGHT)
            circle.rotate(angle, axis=UP)
            circle.move_to(self.axes.c2p(0, 0, 0))
            sphere_circles.add(circle)
        # Equator (highlighted)
        equator = Circle(radius=2, color=BLUE_D, stroke_width=1.5, stroke_opacity=0.6)
        equator.rotate(PI / 2, axis=RIGHT)
        equator.move_to(self.axes.c2p(0, 0, 0))
        sphere_circles.add(equator)
        self.sphere = sphere_circles

        # Primary vector arrow
        p = PRIMARY_VECTOR
        self.primary_arrow = Arrow3D(
            start=self.axes.c2p(0, 0, 0),
            end=self.axes.c2p(*p),
            color=PRIMARY_COLOR,
        )

        # Label for primary vector
        self.primary_label = MathTex(r"\mathbf{w}", color=PRIMARY_COLOR, font_size=36)
        # Position label at the tip
        self.primary_label.move_to(
            self.axes.c2p(*(p * 1.25))
        )
        self.add_fixed_orientation_mobjects(self.primary_label)

        # Animate
        self.play(Create(self.axes), run_time=1)
        self.play(Create(self.sphere), run_time=1.5)
        self.play(Create(self.primary_arrow), FadeIn(self.primary_label))
        self.wait(1)

        # Slow rotation to show 3D
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(2)
        self.stop_ambient_camera_rotation()
        self.wait(0.5)

        # HUD: running total (fixed to screen)
        self.running_total = 0.0
        self.total_text = MathTex(
            r"\text{ReLU sum} = 0.000",
            font_size=32,
            color=WHITE,
        ).to_corner(UL, buff=0.5)
        self.add_fixed_in_frame_mobjects(self.total_text)
        self.play(FadeIn(self.total_text))

    def segment_dot_products(self):
        for i, rv in enumerate(RANDOM_VECTORS):
            self._animate_single_dot_product(i, rv)

    def _animate_single_dot_product(self, index, vec):
        dot_val = float(np.dot(PRIMARY_VECTOR, vec))
        relu_val = max(0.0, dot_val)
        is_positive = dot_val > 0

        vec_color = POSITIVE_COLOR if is_positive else NEGATIVE_COLOR

        # Draw the random vector
        arrow = Arrow3D(
            start=self.axes.c2p(0, 0, 0),
            end=self.axes.c2p(*vec),
            color=vec_color,
        )

        # Vector label
        vec_label = MathTex(
            rf"\mathbf{{x}}_{{{index + 1}}}",
            color=vec_color,
            font_size=30,
        )
        vec_label.move_to(self.axes.c2p(*(vec * 1.2)))
        self.add_fixed_orientation_mobjects(vec_label)

        self.play(Create(arrow), FadeIn(vec_label), run_time=0.6)

        # Show projection line onto primary vector (dot product visualization)
        proj_length = dot_val
        proj_point = PRIMARY_VECTOR * proj_length
        proj_line = DashedLine(
            self.axes.c2p(*vec),
            self.axes.c2p(*proj_point),
            color=GRAY,
            stroke_width=2,
        )
        proj_dot = Dot3D(
            self.axes.c2p(*proj_point),
            color=vec_color,
            radius=0.05,
        )

        self.play(Create(proj_line), FadeIn(proj_dot), run_time=0.5)

        # Info text (fixed to frame)
        dot_str = f"{dot_val:+.3f}"
        relu_str = f"{relu_val:.3f}"
        info = MathTex(
            rf"\mathbf{{w}} \cdot \mathbf{{x}}_{{{index + 1}}} = {dot_str}"
            rf"\quad \Rightarrow \quad \text{{ReLU}} = {relu_str}",
            font_size=28,
            color=vec_color,
        ).to_edge(DOWN, buff=0.5)
        self.add_fixed_in_frame_mobjects(info)
        self.play(FadeIn(info, shift=UP * 0.3), run_time=0.4)
        self.wait(0.8)

        # Update running total
        self.running_total += relu_val
        new_total_text = MathTex(
            rf"\text{{ReLU sum}} = {self.running_total:.3f}",
            font_size=32,
            color=WHITE,
        ).to_corner(UL, buff=0.5)
        self.add_fixed_in_frame_mobjects(new_total_text)

        if is_positive:
            # Flash green to emphasize positive contribution
            self.play(
                Transform(self.total_text, new_total_text),
                arrow.animate.set_color(WHITE),
                run_time=0.5,
            )
            self.play(arrow.animate.set_color(POSITIVE_COLOR), run_time=0.3)
        else:
            # Fade out to emphasize zero contribution
            self.play(
                Transform(self.total_text, new_total_text),
                arrow.animate.set_opacity(0.3),
                run_time=0.5,
            )

        self.wait(0.3)

        # Clean up: fade out the info text, projection, and dim the vector
        self.remove_fixed_in_frame_mobjects(info)
        self.play(
            FadeOut(info),
            FadeOut(proj_line),
            FadeOut(proj_dot),
            arrow.animate.set_opacity(0.2),
            vec_label.animate.set_opacity(0.2),
            run_time=0.4,
        )

    def segment_summary(self):
        self.wait(0.5)

        # Final result box
        result = MathTex(
            rf"\sum_{{i=1}}^{{10}} \text{{ReLU}}(\mathbf{{w}} \cdot \mathbf{{x}}_i)"
            rf"= {self.running_total:.3f}",
            font_size=36,
            color=ACCENT_COLOR,
        ).to_edge(DOWN, buff=1.0)
        box = SurroundingRectangle(result, color=ACCENT_COLOR, buff=0.2)
        self.add_fixed_in_frame_mobjects(result, box)

        self.play(FadeIn(result, shift=UP * 0.3), Create(box), run_time=1)

        # Final rotation
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # Fade everything out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.5)


class DotProductReLU2D(Scene):
    """2D companion scene showing the ReLU accumulation as a bar chart."""

    def construct(self):
        title = Text("ReLU Dot Product Accumulation", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title))

        # Compute all dot products and ReLU values
        dot_vals = [float(np.dot(PRIMARY_VECTOR, rv)) for rv in RANDOM_VECTORS]
        relu_vals = [max(0.0, d) for d in dot_vals]

        # Create bar chart
        bar_width = 0.6
        max_abs = max(abs(d) for d in dot_vals) * 1.2
        bars = VGroup()
        labels = VGroup()
        dot_labels = VGroup()

        for i, (dv, rv) in enumerate(zip(dot_vals, relu_vals)):
            x_pos = (i - 4.5) * (bar_width + 0.3)
            is_pos = dv > 0

            # Full dot product bar (dimmed)
            bar_height = (abs(dv) / max_abs) * 3.0
            direction = UP if dv > 0 else DOWN
            full_bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_color=GRAY,
                fill_opacity=0.2,
                stroke_color=GRAY,
                stroke_width=1,
            )
            full_bar.move_to(ORIGIN + RIGHT * x_pos, aligned_edge=DOWN if dv > 0 else UP)

            # ReLU bar (highlighted)
            if is_pos:
                relu_bar = Rectangle(
                    width=bar_width,
                    height=bar_height,
                    fill_color=POSITIVE_COLOR,
                    fill_opacity=0.7,
                    stroke_color=POSITIVE_COLOR,
                    stroke_width=2,
                )
                relu_bar.move_to(ORIGIN + RIGHT * x_pos, aligned_edge=DOWN)
            else:
                relu_bar = Rectangle(
                    width=bar_width,
                    height=0.02,
                    fill_color=NEGATIVE_COLOR,
                    fill_opacity=0.5,
                    stroke_color=NEGATIVE_COLOR,
                    stroke_width=1,
                )
                relu_bar.move_to(ORIGIN + RIGHT * x_pos)

            bars.add(full_bar, relu_bar)

            # x-axis label
            label = MathTex(rf"\mathbf{{x}}_{{{i+1}}}", font_size=20)
            label.next_to(full_bar, DOWN if dv > 0 else UP, buff=0.15)
            labels.add(label)

            # Value label
            val_label = Text(f"{dv:+.2f}", font_size=14, color=POSITIVE_COLOR if is_pos else NEGATIVE_COLOR)
            val_label.next_to(relu_bar if is_pos else full_bar, UP if dv > 0 else DOWN, buff=0.1)
            dot_labels.add(val_label)

        # Zero line
        zero_line = Line(LEFT * 6, RIGHT * 6, color=WHITE, stroke_width=1)

        # Animate segment by segment
        self.play(Create(zero_line))

        for i in range(10):
            self.play(
                FadeIn(bars[2 * i]),
                FadeIn(bars[2 * i + 1]),
                FadeIn(labels[i]),
                FadeIn(dot_labels[i]),
                run_time=0.4,
            )

        self.wait(0.5)

        # Show sum
        total = sum(relu_vals)
        sum_text = MathTex(
            rf"\sum \text{{ReLU}}(\mathbf{{w}} \cdot \mathbf{{x}}_i) = {total:.3f}",
            font_size=32,
            color=ACCENT_COLOR,
        ).to_edge(DOWN, buff=0.5)
        box = SurroundingRectangle(sum_text, color=ACCENT_COLOR, buff=0.15)
        self.play(FadeIn(sum_text), Create(box))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1)
