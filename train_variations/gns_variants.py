"""train_variations/gns_variants.py

Controllers that adjust training behavior based on the measured
Gradient Noise Scale (GNS).  The design mirrors the optimizer
variant helpers and exposes a dictionary so that different feedback
strategies can be selected from the command line.

Each controller is initialized with the full ``args`` namespace and
provides an ``update(trainer)`` method.  The trainer instance carries
state such as the current GNS estimate, batch size and learning rate.
The controller is free to manipulate ``trainer.args`` to implement a
particular feedback rule.

The variants below are inspired by the analysis in
"An Empirical Model of Large-Batch Training" (McCandlish et al. 2018)
and "Efficient and Approximate Per-Example Gradient Norms for Gradient
Noise Scale" (Gray et al. 2023).  They demonstrate a spectrum of
schemes for keeping the GNS near a desired target and for coupling the
batch size with the learning rate to maintain a stable training
"temperature".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _BaseController:
    args: any

    def update(self, trainer) -> None:  # pragma: no cover - interface
        """Adjust training hyper-parameters based on trainer.gns."""
        raise NotImplementedError


@dataclass
class _NullController(_BaseController):
    """No-op controller – leaves the batch size untouched."""

    def update(self, trainer) -> None:  # pragma: no cover - trivial
        return


@dataclass
class _ProportionalController(_BaseController):
    """Simple proportional controller.

    Replicates the previous heuristic: if the measured GNS falls below
    the target, increase the batch size by ``gns_batch_pct`` (capped by
    ``gns_max_batch``); if it rises above the target, decrease it by the
    same percentage.
    """

    def update(self, trainer) -> None:
        gns = trainer.gns
        target = self.args.gns_target
        if gns is None or target is None:
            return
        pct = self.args.gns_batch_pct
        max_batch = self.args.gns_max_batch
        if gns < target and trainer.args.batch_size < max_batch:
            trainer.args.batch_size = math.ceil(trainer.args.batch_size * (1.0 + pct))
        elif gns > target:
            trainer.args.batch_size = max(1, math.ceil(trainer.args.batch_size * (1.0 - pct)))


@dataclass
class _DoublingController(_BaseController):
    """Binary doubling/halving strategy.

    When GNS is below the target, double the batch size until
    ``gns_max_batch`` is reached.  When it is above the target, halve
    the batch size.  This mirrors the coarse adjustment procedure used
    in a number of large batch-size studies.
    """

    def update(self, trainer) -> None:
        gns = trainer.gns
        target = self.args.gns_target
        if gns is None or target is None:
            return
        if gns < target and trainer.args.batch_size < self.args.gns_max_batch:
            trainer.args.batch_size = min(self.args.gns_max_batch, trainer.args.batch_size * 2)
        elif gns > target and trainer.args.batch_size > 1:
            trainer.args.batch_size = max(1, trainer.args.batch_size // 2)


@dataclass
class _PIDController(_BaseController):
    """PID style controller for smoother convergence to the target GNS."""

    kp: float = 0.1
    ki: float = 0.0
    kd: float = 0.0
    _integral: float = 0.0
    _prev_error: float | None = None

    def update(self, trainer) -> None:
        gns = trainer.gns
        target = self.args.gns_target
        if gns is None or target is None:
            return
        error = target - gns
        self._integral += error
        derivative = 0.0 if self._prev_error is None else error - self._prev_error
        self._prev_error = error
        adj = self.kp * error + self.ki * self._integral + self.kd * derivative
        if adj == 0:
            return
        new_batch = trainer.args.batch_size * (1.0 + adj)
        new_batch = int(max(1, min(self.args.gns_max_batch, new_batch)))
        trainer.args.batch_size = new_batch


@dataclass
class _SqrtRatioController(_BaseController):
    """Scale batch size according to the square root ratio of target to GNS.

    The analysis of McCandlish et al. (2018) shows that training speed
    drops roughly as ``1 + B_noise/B``.  Solving for the batch size that
    keeps this factor constant yields ``B \propto \sqrt{target / gns}``.
    This controller therefore rescales the batch size in proportion to
    the square root of the ratio between the desired target and the
    current gradient noise scale.  It reacts quickly while avoiding the
    large oscillations of pure proportional control.
    """

    def update(self, trainer) -> None:
        gns = trainer.gns
        target = self.args.gns_target
        if gns is None or target is None or gns <= 0:
            return
        factor = math.sqrt(target / gns)
        if factor == 1.0:
            return
        new_batch = int(trainer.args.batch_size * factor)
        new_batch = max(1, min(self.args.gns_max_batch, new_batch))
        trainer.args.batch_size = new_batch


@dataclass
class _TemperatureController(_ProportionalController):
    """Proportional rule that also rescales the learning rate.

    The learning-rate to batch-size ratio acts as a proxy for the
    training *temperature* described in the GNS literature.  Keeping the
    ratio constant while changing the batch size helps to maintain a
    similar optimisation dynamics when parallelising across more
    hardware.
    """

    def update(self, trainer) -> None:
        old_batch = trainer.args.batch_size
        super().update(trainer)
        if trainer.args.batch_size != old_batch:
            scale = trainer.args.batch_size / float(old_batch)
            trainer.args.learning_rate *= scale


# Public dictionary -----------------------------------------------------------

gns_feedback_dictionary = {
    "none": lambda args: _NullController(args),
    "proportional": lambda args: _ProportionalController(args),
    "doubling": lambda args: _DoublingController(args),
    "pid": lambda args: _PIDController(args, kp=args.gns_pid_kp, ki=args.gns_pid_ki, kd=args.gns_pid_kd),
    "sqrt_ratio": lambda args: _SqrtRatioController(args),
    "temperature": lambda args: _TemperatureController(args),
}

