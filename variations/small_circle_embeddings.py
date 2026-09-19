"""Tied virtual tokens on a learnable small circle, with no token lookup parameters.

x(t) = R [a*n + sqrt(1-a*a) (u*cos(2*pi*t) + v*sin(2*pi*t))].
The three frame columns n, u, v are orthonormal. Integer inputs index the
equally spaced vocabulary; floating inputs are continuous phases (one turn = 1).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SmallCircleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, radius=None,
                 offset_init=0.5, learn_offset=True):
        super().__init__()
        if embedding_dim < 3 or num_embeddings < 2:
            raise ValueError("small_circle needs dimension >= 3 and vocabulary >= 2")
        radius = math.sqrt(embedding_dim) if radius is None else float(radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("small_circle radius must be finite and positive")
        if not 0 <= offset_init < 1:
            raise ValueError("circle_offset_init must be in [0, 1)")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.learn_offset = learn_offset
        # Initialize away from the rank-deficient frames where QR derivatives fail.
        frame, _ = torch.linalg.qr(torch.randn(embedding_dim, 3), mode="reduced")
        self.raw_frame = nn.Parameter(frame)
        self.register_buffer("sphere_radius", torch.tensor(radius))
        self.register_buffer("phases", torch.arange(num_embeddings) / num_embeddings)
        if learn_offset:
            # Stay below 1 even in floating point; the loop never collapses exactly.
            if not 0 < offset_init < 1 - 1e-4:
                raise ValueError("a learned offset must start in (0, 0.9999)")
            p = offset_init / (1 - 1e-4)
            self.offset_logit = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        else:
            self.register_buffer("fixed_offset", torch.tensor(float(offset_init)))

    def geometry(self):
        # QR has no half/bfloat16 CPU implementation. Keep this tiny calculation
        # in at least float32, including under autocast.
        work = self.raw_frame if self.raw_frame.dtype == torch.float64 else self.raw_frame.float()
        with torch.autocast(device_type=work.device.type, enabled=False):
            frame, triangular = torch.linalg.qr(work, mode="reduced")
            sign = torch.where(triangular.diagonal() < 0, -1.0, 1.0)
            frame = frame * sign  # remove QR's arbitrary column sign convention
            n, u, v = frame.unbind(dim=1)
            a = ((1 - 1e-4) * self.offset_logit.sigmoid()
                 if self.learn_offset else self.fixed_offset)
            radius = self.sphere_radius * torch.sqrt(1 - a.square())
            return {"center": self.sphere_radius * a * n, "n": n, "u": u,
                    "v": v, "radius": radius, "offset": a,
                    "sphere_radius": self.sphere_radius}

    def embed_phase(self, phase):
        geometry = self.geometry()
        phase = torch.as_tensor(phase, device=self.raw_frame.device,
                                dtype=geometry["u"].dtype)
        theta = (2 * math.pi * phase).unsqueeze(-1)
        result = geometry["center"] + geometry["radius"] * (
            theta.cos() * geometry["u"] + theta.sin() * geometry["v"])
        return result.to(self.raw_frame.dtype)

    @property
    def weight(self):
        """A differentiable virtual table, used for BOTH embedding and LM head."""
        return self.embed_phase(self.phases)

    @property
    def bias(self):
        return None

    def forward(self, tokens):
        if tokens.is_floating_point():
            return self.embed_phase(tokens)
        return F.embedding(tokens, self.weight)

    def decode_phase(self, hidden):
        """Continuous phase maximizing hidden @ x(t); no extra output projection.

        atan2(0, 0) is conventionally returned as zero; at a zero in-plane
        projection every phase has equal score and that answer is uninformative.
        """
        geometry = self.geometry()
        angle = torch.atan2(hidden @ geometry["v"], hidden @ geometry["u"])
        return (angle / (2 * math.pi)).remainder(1)
