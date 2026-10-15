"""
Zero-order optimizers for OLMo training (MeZO / ZO-SGD family first).

Adapted from ``llm-base-zero-order`` / the previous OLMo-Zero-Order-Optimization fork.
``ZeroOrderOptimizer`` is a marker so ``Trainer`` can skip backward and grad clip.
SPSA family: MeZO, ZoAdam, ZoSignSgd; LOZO; SubZeroPlus (Haar low-rank subspace +
multi-query + in-subspace Adam, arXiv:2608.15665); ZOMuon (subspace + Newton–Schulz);
KronZO (Kronecker-factored probes + criterion-driven update); HiZOO (diagonal-Hessian
pre-conditioner); AGZO / AGZOKron (activation-guided probes, arXiv:2601.17261).
``_HessianPreconditioner`` is the shared HiZOO curvature mix-in.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer


class ZeroOrderOptimizer(Optimizer):
    """Marker base class; the trainer uses this to skip backward and gradient clipping."""

    pass


class VectorSampler:
    def __init__(self, sampler_type: str = "standard_normal"):
        if sampler_type not in ("standard_normal", "lp_sphere"):
            raise ValueError(
                f"Unknown sampler_type: {sampler_type!r}. Choose 'standard_normal' or 'lp_sphere'."
            )
        self.sampler_type = sampler_type

    def sample(self, shape: torch.Size, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        if self.sampler_type == "standard_normal":
            return torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
        x = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
        norm = x.norm()
        return x / norm if norm > 0 else x


class MeZO(ZeroOrderOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            momentum=momentum,
            num_pert_samples=num_pert_samples,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        """Seed each device's generator once before iterating parameters.

        Calling manual_seed inside the parameter loop would reset the RNG state
        for every parameter, causing all parameters of the same shape to receive
        the identical perturbation vector.  Seeding once here lets the generator
        advance sequentially through all parameters so each gets an independent z.
        Both _perturb and _apply_update call this before their loops, so they
        reproduce the same per-parameter z from the same seed.
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar estimate ``(f⁺ − f⁻) / (2ε)``
            (or ``(f⁺ − f⁻) / ε`` for one_side mode).  Indicates the
            magnitude of the loss difference along the perturbation direction.
        grad_est_norm
            Global L2 norm of the gradient estimate vector
            ``z * projected_grad / ε`` concatenated across all parameters
            (before lr scaling and weight decay).  Comparable to a gradient
            norm in first-order training.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the raw probe
            directions ``z`` (same samples as in the estimate).  For MeZO this
            equals ``|projected_grad| / ε`` and removes the ``√(num_params)``
            scale from ``||z||₂``.  Not the norm of a direction-normalized
            ``ĝ`` (that would be trivially 1).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    def _probe_z(self, p: torch.Tensor, gen: torch.Generator, z_seed: int) -> torch.Tensor:
        """The probe direction drawn for this parameter.

        ``_perturb`` and ``_apply_update`` must rebuild byte-identical draws from the same
        seed, so both go through here. A subclass may return any direction it likes as
        long as it consumes the generator identically on every call.
        """
        return self.vector_sampler.sample(p.shape, p.device, gen)

    def _active(self, p: torch.Tensor) -> bool:
        """Whether this parameter takes part in the current step.

        Always ``True`` for MeZO — the hook exists so a subclass can restrict the step to a
        subset of tensors without duplicating ``_perturb`` / ``_apply_update`` (see ``LeZO``).
        Every loop that draws ``z`` honours it, so perturb and update stay in sync: the
        shared per-device generator advances over exactly the same parameters in both.
        """
        return True

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            self._perturb(seed_i, scaling_factor=+1.0)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item() / 2.0
                self._perturb(seed_i, scaling_factor=+1.0)
            else:
                self._perturb(seed_i, scaling_factor=-1.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item()

            samples.append((seed_i, scalar))

        self._apply_update(samples)
        self._last_samples = samples  # fo_cosine_metrics rebuilds the step from these
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        return first_loss  # type: ignore[return-value]

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Cosine between the update direction and the true gradient.

        Must be called right after ``step()`` with ``p.grad`` holding ∇L(θ_k).

        The direction reported is the aggregated SPSA estimate ``Σ scalar_i·z_i`` — what
        the optimizer actually steps along — not the raw probe. That distinction matters:
        a probe's signed cosine with the gradient is symmetric about zero (``z`` and
        ``−z`` are equally likely), so its mean is ~0 and says nothing, whereas the
        estimate carries the sign the finite difference recovered and its cosine is the
        effective alignment of the step. For isotropic MeZO this lands near ``√(2/πd)``,
        the floor every probe-learning method is trying to beat.

        Key name matches the AGZO family so both land in the same W&B panel.
        """
        samples = getattr(self, "_last_samples", None)
        if not samples:
            return {}
        # Rebuild the estimate exactly as _apply_update does. Costs one model-sized
        # buffer, but this runs only on diagnostic steps, which already pay for a full
        # backward -- so it does not change the run's peak memory.
        acc: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and self._active(p) and p.grad is not None:
                    acc[id(p)] = torch.zeros_like(p, dtype=torch.float32)
        for seed_i, scalar_i in samples:
            self._reset_generators(seed_i)
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad or not self._active(p):
                        continue
                    # Draw for every active parameter, gradient or not: the shared
                    # generator must advance over the same sequence it did during the
                    # step, or the rebuilt direction is not the one that was taken.
                    z = self._probe_z(p, self._get_generator(p.device), seed_i)
                    if id(p) in acc:
                        acc[id(p)].add_(z.float(), alpha=scalar_i)
        dot = norm_u = norm_g = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                u = acc.get(id(p))
                if u is None:
                    continue
                g = p.grad.float()
                dot += (u * g).sum().item()
                norm_u += u.pow(2).sum().item()
                norm_g += g.pow(2).sum().item()
        if norm_g == 0.0 or norm_u == 0.0:
            return {}
        return {"cosine_qrge": dot / (math.sqrt(norm_g * norm_u) + 1e-12)}


    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._reset_generators(z_seed)
        for group in self.param_groups:
            eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad or not self._active(p):
                    continue
                gen = self._get_generator(p.device)
                z = self._probe_z(p, gen, z_seed)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)

        # Float32 accumulators: sum_i scalar_i * z_i / eps, averaged over n.
        accumulators: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and self._active(p):
                    accumulators[id(p)] = torch.zeros_like(p, dtype=torch.float32)

        z_sum_sq = 0.0
        for seed_i, scalar_i in samples:
            self._reset_generators(seed_i)
            for group in self.param_groups:
                eps = group["zo_eps"]
                for p in group["params"]:
                    if not p.requires_grad or not self._active(p):
                        continue
                    gen = self._get_generator(p.device)
                    z = self._probe_z(p, gen, seed_i)
                    z_sum_sq += z.float().norm().item() ** 2
                    accumulators[id(p)].add_(z.float(), alpha=scalar_i / eps)

        grad_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if not p.requires_grad or not self._active(p):
                    continue
                grad_est = accumulators[id(p)].div_(n)
                grad_sum_sq += grad_est.norm().item() ** 2
                if weight_decay != 0.0:
                    grad_est.add_(p.data.float(), alpha=weight_decay)
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad_est.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad_est)
                        grad_est = buf
                p.data.add_(grad_est.to(p.dtype), alpha=-lr)
        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")


class LeZO(MeZO):
    """LeZO — MeZO with layer-wise sparsity (arXiv:2410.09823, "Simultaneous Computation and
    Memory Efficient Zeroth-Order Optimizer for Fine-Tuning Large Language Models").

    Plain MeZO perturbs and updates *every* parameter on every step, and the paper measures
    that this perturb/update bookkeeping eats >50% of MeZO's wall time. LeZO freezes a
    fraction of the "layers" on each step instead:

      1. Pick an active subset of ``(1 - layer_sparsity)`` of the trainable tensors.
      2. Perturb and update only those — SPSA and the ZO-SGD step use the *same* subset, so
         ``z`` is never drawn for a frozen tensor; the rest keep their values untouched.
      3. Rotate the subset with the step count, so over training every tensor is trained.
         This stays full-parameter fine-tuning, not a frozen-layer scheme.

    Two wins: less perturb/update work per step (the paper reports >3x speedup at ρ=0.75),
    and a probe confined to ``d_active < d``, which lifts the per-step ZO/FO alignment
    (~``1/√d_active``).

    The selection mechanism is deliberately **identical to ``AGZO._select_active``** (and to
    ``KronZO``'s): a ``torch.randperm`` seeded by ``_step_count`` on CPU over the individual
    parameter tensors. Two consequences worth knowing:

    * It is seeded rather than truly random, so the subset is identical on every DDP rank
      with no communication — the contract every ZO optimizer in this file relies on.
      Statistically it is still a uniform random subset, just reproducible.
    * The unit is **one parameter tensor**, whereas the paper's unit is a whole transformer
      block (its ``N`` counts blocks, and embeddings / final norm / LM head sit outside
      them). Keeping the repo's granularity is what makes ``lezo``, ``kronzo_layer_sparsity``
      and ``agzo_layer_sparsity`` directly comparable to each other.

    Weight decay and momentum do not run on frozen tensors either — a frozen tensor is fully
    untouched for that step. The paper does not discuss this.

    ``layer_sparsity == 0`` degenerates to exactly plain ``MeZO``: every tensor stays active.

    Forward-pass cost is unchanged — ``2`` per probe sample (two_side), the model still runs
    end to end; only the perturb/update tensor work shrinks. Everything else (momentum,
    ``num_pert_samples``, the sampler, the metrics) is inherited from ``MeZO``.

    Note on lr: the paper finds that higher sparsity wants a *higher* learning rate (found
    by grid search, no closed-form rescaling). Re-tune ``learning_rate`` when you change
    ``layer_sparsity``.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        layer_sparsity: float = 0.0,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        num_pert_samples: int = 1,
    ):
        if not (0.0 <= layer_sparsity < 1.0):
            raise ValueError(f"layer_sparsity must be in [0, 1): {layer_sparsity}")
        super().__init__(
            params,
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            momentum=momentum,
            vector_sampling_type=vector_sampling_type,
            num_pert_samples=num_pert_samples,
        )
        self.defaults["layer_sparsity"] = layer_sparsity
        self._step_count = 0
        # Ids of the tensors active (perturbed + updated) this step. Empty until
        # _select_active runs; with layer_sparsity == 0 it is always the full set.
        self._active_wids: set[int] = set()
        self._active_frac: float = 1.0

    def _flat_params(self):
        """Same (group, param, index) walk the AGZO / KronZO families use."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _select_active(self) -> None:
        """Pick which parameter tensors this step perturbs and updates.

        Same rule as ``AGZO._select_active``: the active subset rotates deterministically
        with ``_step_count`` (torch.randperm on CPU, identical on every DDP rank — no
        communication), so over many steps every tensor is trained, but each step touches
        only a ``(1 - layer_sparsity)`` fraction. ``layer_sparsity == 0`` keeps every tensor
        active (plain MeZO).
        """
        params = [p for _g, p, _i in self._flat_params() if p.requires_grad]
        n_total = len(params)
        sparsity = self.defaults["layer_sparsity"]
        if sparsity <= 0.0 or n_total == 0:
            self._active_wids = {id(p) for p in params}
            self._active_frac = 1.0
            return
        n_keep = max(1, round(n_total * (1.0 - sparsity)))
        gen = torch.Generator()
        gen.manual_seed((self._step_count * 2_654_435_761 + 12_345) % (2**31))
        perm = torch.randperm(n_total, generator=gen).tolist()
        self._active_wids = {id(params[perm[i]]) for i in range(n_keep)}
        self._active_frac = n_keep / n_total

    def _active(self, p: torch.Tensor) -> bool:
        return id(p) in self._active_wids

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        # Fix the active subset before MeZO's probe/update loops — both must see the same
        # set, otherwise the shared per-device RNG would desynchronize between them.
        self._select_active()
        loss = super().step(closure, z_seed=z_seed)
        self._step_count += 1
        self._last_metrics["layer_active_frac"] = self._active_frac
        return loss


_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)  # E|ĝ·ξ̂| for an isotropic draw

# Per-seed reduction of the probe: z = a·∇L + b·ξ, with ‖∇L‖ and z·∇L carried along so
# neither the update magnitude nor the descent sign needs a second pass over parameters.
_Probe = namedtuple("_Probe", "a b norm_g dot_zg")


class OracleProbe(MeZO):
    """MeZO whose probe direction has a *prescribed* cosine with the true gradient.

    This is a measuring instrument, not a training method.  It needs a real
    forward+backward every step, so it has neither the memory profile nor the
    forward-pass budget of a zero-order optimizer and must never be reported as one.

    It exists to answer one question that no ZO run can answer on its own: **how much
    probe alignment does a sampler have to deliver before the loss curve actually
    moves?**  Historical AGZO runs sit at ``cosine_qrge ≈ 3e-4`` and a learned sampler
    is only worth building if the loss is sensitive to alignment somewhere near that
    value.  Sweeping ``target_cosine`` with everything else held fixed turns that from
    a hunch into a number.

    Construction.  With ``ĝ = ∇L/‖∇L‖`` and an isotropic draw ``ξ``, decompose ξ into
    its component along ĝ and the rest, and recombine at the requested angle::

        ξ_⊥ = ξ − (ξ·ĝ)ĝ
        z   = ‖ξ‖ · ( c·ĝ + √(1−c²)·ξ̂_⊥ )

    so ``cos(z, ∇L) = c`` exactly and ``‖z‖ = ‖ξ‖``.  Matching the norm to the Gaussian
    draw matters: the SPSA update is ``(f⁺−f⁻)/(2ε) · z / ε``, which scales with ``‖z‖``,
    so a unit-norm probe would silently shrink the step by ``√d`` and confound the
    alignment sweep with a learning-rate sweep.  At ``c = 0`` this reduces to plain
    MeZO (ξ minus a vanishing component), which is the control arm of the ladder.

    Why the update is rescaled (``update_rule='matched'``, the default).  The raw SPSA
    estimate is ``ĝ = ((f⁺−f⁻)/2ε)·z ≈ (∇L·z)·z``, and ``∇L·z = c‖∇L‖‖z‖``.  Because
    ``‖z‖ = √d``, the estimate's magnitude grows by a factor of ``c√d`` across the
    ladder — at d≈1.5e8 and c=0.03 that is ~370× — so a fixed learning rate makes the
    sweep a *step-size* sweep wearing a cosine costume.  Measured, not theorised: the
    first c=0.03 smoke diverged from 11.5 to 128 nats in eight steps.

    Why the sign is taken exactly (``sign_source='exact'``, the default).  The two probe
    forwards exist only to recover ``sign(∇L·z)``, which an oracle already knows.  Keeping
    them is actively harmful: at ``ε‖z‖ ≈ 15`` the finite difference is a chord rather than
    a derivative, and its error grows with ``c`` because the aligned part of the
    displacement is ``c·ε‖z‖`` and the steepest direction is also the most curved.  With
    the difference retained, c=0.10 trained 1.28 nats *worse* than c=0.03 at two learning
    rates — a monotone-backwards ladder.  Exact mode removes ε from the instrument (it
    cancels), drops two forwards per step, and reports ``f(θ)`` instead of ``f(θ+εz)``.
    ``spsa`` restores the measured sign, which is what a real ZO method must live with and
    is the right setting for studying the ε interaction itself.

    ``matched`` keeps the sign the two-point measurement produced and replaces the
    magnitude with the one an isotropic probe would have given, ``E|∇L·z| =
    √(2/π)·‖∇L‖·‖z‖``.  Every rung then takes the same size of step and only the
    *direction* varies, which is the whole point of the ladder.  At c=0 this is MeZO
    with the magnitude noise removed.  ``spsa`` restores the raw estimator.
    """

    def __init__(
        self,
        *args,
        target_cosine: float = 0.0,
        update_rule: str = "matched",
        sign_source: str = "exact",
        **kwargs,
    ):
        if not -1.0 <= target_cosine <= 1.0:
            raise ValueError(f"target_cosine must be in [-1, 1], got {target_cosine}")
        if update_rule not in ("matched", "spsa"):
            raise ValueError(f"update_rule must be 'matched' or 'spsa', got {update_rule!r}")
        if sign_source not in ("exact", "spsa"):
            raise ValueError(f"sign_source must be 'exact' or 'spsa', got {sign_source!r}")
        super().__init__(*args, **kwargs)
        self.target_cosine = float(target_cosine)
        self.update_rule = update_rule
        self.sign_source = sign_source
        self._reference_loss: Optional[torch.Tensor] = None
        # (A, B, ‖∇L‖) per probe seed for the current step; z = A·∇L + B·ξ. Cleared by
        # step() because ∇L moves with the parameters.
        self._ab_cache: dict[int, _Probe] = {}

    # ------------------------------------------------------------------
    # Probe construction
    # ------------------------------------------------------------------

    def _ensure_ab(self, z_seed: int) -> _Probe:
        """Reduce the construction above to two scalars, in one pass over the parameters.

        ``‖ξ_⊥‖`` has a closed form given the three accumulators, so no second pass and
        no stored ξ are needed::

            ‖ξ_⊥‖² = ‖ξ‖² − (ξ·∇L)²/‖∇L‖²

        Under DDP every rank holds a full replica, the gradients are all-reduced by the
        backward, and ξ comes from a seed the trainer synchronises — so all ranks reach
        identical scalars with no extra communication.
        """
        cached = self._ab_cache.get(z_seed)
        if cached is not None:
            return cached

        self._reset_generators(z_seed)
        norm_g2 = norm_xi2 = dot = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad or not self._active(p):
                    continue
                gen = self._get_generator(p.device)
                # Draw ξ even where there is no gradient: the generator must advance over
                # exactly the same parameters here as in _probe_z, or the probe rebuilt
                # during _apply_update will not match the one used to perturb.
                xi = self.vector_sampler.sample(p.shape, p.device, gen).float()
                norm_xi2 += xi.pow(2).sum().item()
                if p.grad is None:
                    continue
                g = p.grad.float()
                norm_g2 += g.pow(2).sum().item()
                dot += (xi * g).sum().item()

        if norm_g2 == 0.0:
            raise RuntimeError(
                "OracleProbe.step() found no gradients. It requires ∇L(θ_k) in .grad on "
                "every step — the trainer runs that backward only when the optimizer is "
                "an OracleProbe. Without it the probe would silently degrade to isotropic "
                "MeZO and the whole cosine ladder would read as a null result."
            )

        c = self.target_cosine
        norm_g = math.sqrt(norm_g2)
        norm_xi = math.sqrt(norm_xi2)
        # Guard the degenerate draw where ξ is (numerically) parallel to ∇L; in the
        # dimensions we train at this never fires, but the sqrt must not go negative.
        perp2 = max(norm_xi2 - dot * dot / norm_g2, 1e-30)
        norm_perp = math.sqrt(perp2)

        s = math.sqrt(max(0.0, 1.0 - c * c))
        b = norm_xi * s / norm_perp
        a = norm_xi * c / norm_g - b * (dot / norm_g2)

        # z·∇L from the same primitives rather than the analytic c·‖z‖·‖∇L‖ -- the point
        # of an instrument is that it measures instead of asserting, and at c=0 the sign
        # has to come out genuinely undetermined rather than assumed positive.
        probe = _Probe(a=a, b=b, norm_g=norm_g, dot_zg=a * norm_g2 + b * dot)
        self._ab_cache[z_seed] = probe
        self._last_metrics["oracle_target_cosine"] = c
        self._last_metrics["oracle_grad_norm"] = norm_g
        return probe

    def _probe_z(self, p: torch.Tensor, gen: torch.Generator, z_seed: int) -> torch.Tensor:
        pr = self._ab_cache[z_seed]  # warmed by step / _apply_update below
        xi = self.vector_sampler.sample(p.shape, p.device, gen)
        z = xi.float().mul_(pr.b)
        if p.grad is not None and pr.a != 0.0:
            z.add_(p.grad.float(), alpha=pr.a)
        return z.to(p.dtype)

    # Both overrides exist for one reason: _ensure_ab walks the parameters and therefore
    # advances the shared generators, so it has to finish *before* the caller seeds them
    # for its own loop. Doing it lazily inside _probe_z would desynchronise the draw.

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._ensure_ab(z_seed)
        super()._perturb(z_seed, scaling_factor)

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        for seed_i, _ in samples:
            self._ensure_ab(seed_i)
        if self.update_rule == "spsa":
            super()._apply_update(samples)
            return
        # Keep the measured sign, swap in the isotropic magnitude. MeZO divides the
        # scalar by group["zo_eps"], so multiplying by the same eps here lands exactly on
        # √(2/π)·‖∇L‖ — _zo_param_groups stamps one zo_eps on every group, so reading it
        # from defaults matches what _apply_update will divide by.
        eps = self.defaults["zo_eps"]
        matched: list[tuple[int, float]] = []
        for seed_i, scalar_i in samples:
            norm_g = self._ab_cache[seed_i].norm_g
            matched.append((seed_i, math.copysign(_SQRT_2_OVER_PI * norm_g * eps, scalar_i)))
        n = len(samples)
        self._last_metrics["oracle_scalar_spsa_abs"] = sum(abs(s) for _, s in samples) / n
        self._last_metrics["oracle_scalar_used_abs"] = sum(abs(s) for _, s in matched) / n
        super()._apply_update(matched)

    def set_reference_loss(self, loss: torch.Tensor) -> None:
        """Hand over the loss from the trainer's first-order pass at θ.

        In ``sign_source='exact'`` no probe forward happens, so there is no perturbed
        loss to report. This one is strictly better as a training curve: MeZO returns
        ``f(θ+εz)``, which carries the probe's own noise, whereas this is ``f(θ)``.
        """
        self._reference_loss = loss

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        # ∇L moves with the parameters, so last step's scalars are stale.
        self._ab_cache.clear()
        self._last_seed = z_seed

        if self.sign_source == "spsa":
            return super().step(closure, z_seed=z_seed)

        # Exact mode. The two probe forwards existed only to recover sign(∇L·z), and we
        # already hold ∇L, so they are pure cost -- and worse than pure cost: at
        # ε‖z‖ ≈ 15 the two-point difference is a chord, not a derivative, and it
        # degrades in proportion to how well-aligned the probe is (the aligned part of
        # the displacement is c·ε‖z‖, and the steepest direction is the most curved).
        # Measured: c=0.10 finished 1.28 nats *worse* than c=0.03 at two learning rates,
        # which would have made the ladder read backwards. Dropping the finite difference
        # removes ε from the instrument entirely -- note it cancels below, since
        # MeZO._apply_update divides the scalar by the same zo_eps.
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
            self._last_seed = z_seed

        eps = self.defaults["zo_eps"]
        samples: list[tuple[int, float]] = []
        for i in range(self.defaults["num_pert_samples"]):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            pr = self._ensure_ab(seed_i)
            mag = _SQRT_2_OVER_PI * pr.norm_g * eps
            # copysign would turn a genuinely undetermined sign (c=0, z ⟂ ∇L) into a
            # descent claim; an orthogonal probe must contribute nothing, and that is
            # exactly what makes c=0 the honest control arm.
            samples.append((seed_i, math.copysign(mag, pr.dot_zg) if pr.dot_zg != 0.0 else 0.0))

        MeZO._apply_update(self, samples)  # already matched; skip this class's override
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / len(samples)

        if self._reference_loss is None:
            raise RuntimeError(
                "OracleProbe(sign_source='exact') has no reference loss. The trainer must "
                "call set_reference_loss() with the first-order pass's loss; without it "
                "there is no forward at θ to report."
            )
        return self._reference_loss

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Realised cosine of the probe with ∇L — a self-check on the construction.

        ``cosine_qrge`` should equal ``target_cosine`` to floating-point noise. It is
        measured rather than asserted because the whole experiment is an argument about
        this number, and an instrument that reports its own setting back is worthless.
        The key name matches the AGZO family so both land in the same W&B panel.
        """
        if not self._ab_cache:
            return {}
        z_seed = next(iter(self._ab_cache))
        pr = self._ab_cache[z_seed]
        a, b = pr.a, pr.b
        self._reset_generators(z_seed)
        dot = norm_z2 = norm_g2 = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad or not self._active(p):
                    continue
                gen = self._get_generator(p.device)
                xi = self.vector_sampler.sample(p.shape, p.device, gen).float()
                z = xi.mul_(b)
                if p.grad is not None:
                    g = p.grad.float()
                    z.add_(g, alpha=a)
                    norm_g2 += g.pow(2).sum().item()
                    dot += (z * g).sum().item()
                norm_z2 += z.pow(2).sum().item()
        if norm_g2 == 0.0 or norm_z2 == 0.0:
            return {}
        return {
            "cosine_qrge": dot / (math.sqrt(norm_g2 * norm_z2) + 1e-12),
            "cosine_target": self.target_cosine,
        }


class ZoAdam(ZeroOrderOptimizer):
    """
    Zeroth-order Adam (AdamW-style): same random-probe SPSA estimate as MeZO, then
    first/second moment tracking and bias-corrected update like AdamW (decoupled weight decay).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        b1, b2 = betas
        if not (0.0 <= b1 <= 1.0 and 0.0 <= b2 <= 1.0):
            raise ValueError(f"betas must be in [0, 1], got {betas}")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            betas=betas,
            eps=eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        """Seed each device's generator once before iterating parameters.

        See MeZO._reset_generators for rationale.
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar ``(f⁺ − f⁻) / (2ε)``.
        grad_est_norm
            Global L2 norm of ``z * projected_grad / ε`` across all parameters
            (raw SPSA estimate, before Adam preconditioning and lr scaling).
        update_norm
            Global L2 norm of the actual Adam-preconditioned update applied
            to the parameters.  Reflects the effect of moment scaling.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the raw ``z`` probes
            (same as in MeZO).  Equals ``|projected_grad| / ε`` for the SPSA
            block before Adam.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            self._perturb(seed_i, scaling_factor=+1.0)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item() / 2.0
                self._perturb(seed_i, scaling_factor=+1.0)
            else:
                self._perturb(seed_i, scaling_factor=-1.0)
                loss_minus = closure()
                scalar = (loss_plus - loss_minus).item()

            samples.append((seed_i, scalar))

        self._apply_update(samples)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        return first_loss  # type: ignore[return-value]

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._reset_generators(z_seed)
        for group in self.param_groups:
            zo_eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * zo_eps)

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)

        # Float32 accumulators: (1/n) * sum_i scalar_i * z_i / eps
        accumulators: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    accumulators[id(p)] = torch.zeros_like(p, dtype=torch.float32)

        z_sum_sq = 0.0
        for seed_i, scalar_i in samples:
            self._reset_generators(seed_i)
            for group in self.param_groups:
                zo_eps = group["zo_eps"]
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    gen = self._get_generator(p.device)
                    z = self.vector_sampler.sample(p.shape, p.device, gen)
                    z_sum_sq += z.float().norm().item() ** 2
                    accumulators[id(p)].add_(z.float(), alpha=scalar_i / zo_eps)

        grad_sum_sq = 0.0
        update_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                g32 = accumulators[id(p)].div_(n)
                grad_sum_sq += g32.norm().item() ** 2

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=torch.device("cpu"))
                    state["exp_avg"] = torch.zeros_like(g32)
                    state["exp_avg_sq"] = torch.zeros_like(g32)

                state["step"] += 1
                step = int(state["step"].item())
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

                bias_c1 = 1.0 - beta1**step
                bias_c2 = 1.0 - beta2**step
                step_size_neg = -lr / bias_c1
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_c2)).add(adam_eps)
                update = (exp_avg * step_size_neg / denom).to(dtype=p.dtype)
                update_sum_sq += update.to(torch.float32).norm().item() ** 2
                p.data.add_(update)

        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")
        self._last_metrics["update_norm"] = math.sqrt(update_sum_sq)

class ZoSignSgd(ZeroOrderOptimizer):
    """ZO-SignSGD (ported from LDSDSignSgd): SPSA probe + sign(g) update.

    Same finite-difference probing as MeZO, but the projected scalar gradient
    g = (f+ − f−)/2ε is replaced by its sign before scaling the update:

        update_i = z_i * sign(g) / ε

    This makes the effective step magnitude insensitive to the absolute size of
    the loss difference, acting like a sign-based first-order method applied to
    the zero-order estimate.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        vector_sampling_type: str = "standard_normal",
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    # ------------------------------------------------------------------
    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics from the last step.

        projected_grad_abs
            |g| where g = (f+ − f−)/2  (raw, before sign compression).
        grad_est_norm
            L2 norm of ``sign(g) * z / ε`` across all params.
        grad_est_norm_per_z_rms
            Ratio to RMS ``||z||₂``; with ``|sign(g)|=1`` equals ``1/ε``; scales
            with ``|g|`` if you compare to raw ``g`` via ``projected_grad_abs/ε``
            only when the sign is non-degenerate.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor],
        z_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        self._perturb(z_seed, scaling_factor=+1.0)
        loss_plus = closure()

        if self.defaults["perturbation_mode"] == "two_side":
            self._perturb(z_seed, scaling_factor=-2.0)
            loss_minus = closure()
            raw_grad = (loss_plus - loss_minus).item() / 2.0
            self._perturb(z_seed, scaling_factor=+1.0)  # restore
        else:
            self._perturb(z_seed, scaling_factor=-1.0)
            loss_minus = closure()
            raw_grad = (loss_plus - loss_minus).item()

        # sign compression of the scalar gradient
        signed_grad = math.copysign(1.0, raw_grad) if raw_grad != 0.0 else 0.0

        self._apply_update(z_seed, signed_grad)
        self._last_metrics["projected_grad_abs"] = abs(raw_grad)
        return loss_plus

    # ------------------------------------------------------------------
    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        self._reset_generators(z_seed)
        for group in self.param_groups:
            eps = group["zo_eps"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, z_seed: int, signed_grad: float) -> None:
        self._reset_generators(z_seed)
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["zo_eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                z = self.vector_sampler.sample(p.shape, p.device, gen)
                z_sum_sq += z.to(torch.float32).norm().item() ** 2
                grad_est = z.mul(signed_grad / eps)
                grad_sum_sq += grad_est.to(torch.float32).norm().item() ** 2
                if weight_decay != 0.0:
                    grad_est.add_(p.data, alpha=weight_decay)
                if momentum != 0.0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = grad_est.clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad_est)
                        grad_est = buf
                p.data.add_(grad_est, alpha=-lr)
        ge = math.sqrt(grad_sum_sq)
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")

class LOZO(ZeroOrderOptimizer):
    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 4,
        step_interval: int = 1,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute value of the SPSA scalar ``(f⁺ − f⁻) / (2ε)``.
        grad_est_norm
            Global L2 norm of the low-rank gradient estimate ``projected_grad * Z``
            across all parameters (before lr scaling).  For 2D params the
            Frobenius norm of ``U @ V.T`` is computed efficiently via the
            ``rank × rank`` Gram matrices; for 1D params the standard L2 norm.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the probe directions
            (``U @ V.T`` Frobenius contribution per 2-D param, ``z`` L2 per 1-D
            param).  For this optimizer equals ``|projected_grad|`` from
            ``step()`` (the scalar already includes the ``1/ε`` factor).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        num_pert = self.defaults["num_pert_samples"]
        samples: list[tuple[int, float]] = []
        first_loss: Optional[torch.Tensor] = None

        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)
            # V is refreshed only on the first sample of a refresh step.
            refresh_v = (i == 0) and (self._step_count % self.defaults["step_interval"] == 0)

            self._perturb(seed_i, scaling_factor=+1.0, refresh_v=refresh_v)
            loss_plus = closure()
            if first_loss is None:
                first_loss = loss_plus

            if self.defaults["perturbation_mode"] == "two_side":
                self._perturb(seed_i, scaling_factor=-2.0, refresh_v=False)
                loss_minus = closure()
                projected_grad = (loss_plus - loss_minus).item() / (2.0 * self.defaults["zo_eps"])
                self._perturb(seed_i, scaling_factor=+1.0, refresh_v=False)
            else:
                self._perturb(seed_i, scaling_factor=-1.0, refresh_v=False)
                loss_minus = closure()
                projected_grad = (loss_plus - loss_minus).item() / self.defaults["zo_eps"]

            samples.append((seed_i, projected_grad))

        self._apply_update(samples)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for _, s in samples) / num_pert
        self._step_count += 1
        return first_loss  # type: ignore[return-value]

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_u(self, p: torch.Tensor, z_seed: int, param_idx: int) -> torch.Tensor:
        rank = self.defaults["rank"]
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + param_idx * 1_000_003) % (2**31))
        return torch.randn(p.shape[0], rank, device=p.device, dtype=p.dtype, generator=gen)

    def _get_z1d(self, p: torch.Tensor, z_seed: int, param_idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + param_idx * 999_983) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)

    def _perturb(self, z_seed: int, scaling_factor: float, refresh_v: bool) -> None:
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            eps = group["zo_eps"]
            if p.dim() == 2:
                rank = group["rank"]
                state = self.state[p]
                if refresh_v or "v" not in state:
                    gen_v = torch.Generator(device=p.device)
                    gen_v.manual_seed((z_seed * 1_000_003 + idx * 999_983) % (2**31))
                    v = torch.randn(p.shape[1], rank, device=p.device, dtype=p.dtype, generator=gen_v)
                    state["v"] = v
                v = state["v"]
                u = self._get_u(p, z_seed, idx)
                p.data.addmm_(u, v.t(), alpha=scaling_factor * eps)
            else:
                z = self._get_z1d(p, z_seed, idx)
                p.data.add_(z, alpha=scaling_factor * eps)

    def _apply_update(self, samples: list[tuple[int, float]]) -> None:
        n = len(samples)
        grad_sum_sq = 0.0
        z_sum_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if p.dim() == 2:
                v = self.state[p]["v"]  # (n_cols, rank), p.dtype
                effective_rank = v.shape[1]
                vgv = v.float().T @ v.float()

                u_acc = torch.zeros(p.shape[0], effective_rank, device=p.device, dtype=torch.float32)
                for seed_i, pg in samples:
                    u_i = self._get_u(p, seed_i, idx).float()
                    u_acc.add_(u_i, alpha=pg)
                    ugu = u_i.T @ u_i
                    z_sum_sq += (ugu @ vgv).trace().item()
                u_acc.div_(n)

                ugu = u_acc.T @ u_acc
                grad_sum_sq += (ugu @ vgv).trace().item()

                p.data.addmm_(u_acc.to(p.dtype), v.t(), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
            else:
                z_acc = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
                for seed_i, pg in samples:
                    z_i = self._get_z1d(p, seed_i, idx).float()
                    z_acc.add_(z_i, alpha=pg)
                    z_sum_sq += z_i.norm().item() ** 2
                z_acc.div_(n)

                grad_sum_sq += z_acc.norm().item() ** 2
                p.data.add_(z_acc.to(p.dtype), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
        ge = math.sqrt(max(grad_sum_sq, 0.0))
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")


# ---------------------------------------------------------------------------
# SubZero+  (arXiv:2608.15665)
# ---------------------------------------------------------------------------

def _haar_orthonormal(
    rows: int, cols: int, device: torch.device, seed: int, correct_signs: bool = True
) -> torch.Tensor:
    """Return a column-orthonormal ``(rows, cols)`` matrix, float32.

    QR of a Gaussian matrix already gives ``Qᵀ Q = I``, but ``torch.linalg.qr``
    does not pin the sign of ``R``'s diagonal, so the column signs of ``Q`` are
    arbitrary among ``2^cols`` choices and ``Q`` is *not* Haar-distributed.
    Multiplying each column by ``sgn(R_ii)`` selects the unique configuration
    with ``R_ii > 0``, which is Haar (Mezzadri, 2007) and hence directionally
    unbiased.  ``correct_signs=False`` reproduces the raw ``torch.linalg.qr``
    behaviour, for ablations.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed % (2**31))
    omega = torch.randn(rows, cols, device=device, dtype=torch.float32, generator=gen)
    q, r = torch.linalg.qr(omega, mode="reduced")
    if correct_signs:
        signs = r.diagonal().sign()
        signs[signs == 0] = 1.0
        q = q * signs
    return q


class _KronDirectional:
    """KronZO's directional selection (GERAD G-2025-44, Alg. 2) for ``SubZeroPlus`` /
    ``AGZOPlus``, where every trial is a full multi-query estimate.

    With ``query_budget = q > 0`` a step makes ``q`` trials.  Trial ``j`` is exactly the
    host's plain step drawn with probe seed ``trial_seed(z_seed, j)``: ``K`` probes
    (``num_queries``) averaged into one estimate and, with ``update_rule="adam"``, turned
    into an Adam step from the *current* moments.  The candidate step of every trial is
    evaluated on the batch; only the best one is applied — if it beats the centre loss and
    the worst of the last ``h`` accepted losses (``h = 0`` disables the window) — and only
    then are its moments committed.  A rejected step leaves weights and moments untouched.
    Trial 0 draws the same probes as the plain step, so ``q = 1`` is the plain step plus
    the acceptance test.  ``q = 0`` disables selection.

    Cost: ``1 + q·(K+1)`` forwards (one_side) or ``1 + q·(2K+1)`` (two_side).  At an equal
    budget, plain averaging of all ``q·K`` probes has the better expected cosine with the
    gradient (``sqrt(qK)`` vs ``sqrt(K)·sqrt(2 ln q)``); what selection buys is that every
    applied step was checked on the loss first.

    The one_side finite difference is always taken against the centre loss ``L(θ)``;
    ``KronZO`` / ``AGZOKron`` use the running best candidate loss there instead.

    Host contract — the host provides:

    * ``_estimate(closure, center, seed, num_queries, two_side, eps) -> (scalars, first)``
      — the K-probe loop of its plain step;
    * ``_compute_directions(scalars, seed, commit) -> dirs`` — averaged estimate and
      update rule; with ``commit=False`` the optimizer state is left untouched;
    * ``_move(dirs, sign)`` — ``θ += sign · lr · lift(coeff)``; ``_decay(dirs)`` —
      decoupled weight decay on the params in ``dirs``.
    """

    _TRIAL_SEED_STRIDE = 1_000_000_007

    def _init_kron(self, query_budget: int, history_length: int) -> None:
        if query_budget < 0:
            raise ValueError(f"Invalid query_budget: {query_budget}")
        if history_length < 0:
            raise ValueError(f"Invalid history_length: {history_length}")
        self._kron_q = query_budget
        # Prefilled with +inf so the first h steps pass the window test (as in KronZO).
        self._window: list[float] = [float("inf")] * history_length
        self._n_steps = 0
        self._n_accepted = 0

    def _trial_seed(self, z_seed: int, j: int) -> int:
        return (z_seed + j * self._TRIAL_SEED_STRIDE) % (2**31)

    def _kron_accepts(self, center: float, loss: float) -> bool:
        """Better than the centre on this batch and not worse than the worst windowed loss."""
        window_max = max(self._window) if self._window else float("inf")
        return loss < center and loss <= window_max

    def _push_window(self, loss: float) -> None:
        """Replace the worst (max) loss in the acceptance window with ``loss``."""
        if not self._window:
            return
        j = max(range(len(self._window)), key=lambda k: self._window[k])
        self._window[j] = loss

    def _kron_select_step(
        self,
        closure,
        center_loss: torch.Tensor,
        z_seed: int,
        num_queries: int,
        two_side: bool,
        eps: float,
    ) -> dict:
        """q trials of K averaged probes each; apply and commit the best one."""
        center = center_loss.item()
        all_scalars: dict = {}
        best: Optional[tuple[int, dict[int, float]]] = None
        l_best = float("inf")
        for j in range(self._kron_q):
            seed = self._trial_seed(z_seed, j)
            scalars, _ = self._estimate(closure, center_loss, seed, num_queries, two_side, eps)
            all_scalars.update({(j, k): c for k, c in scalars.items()})

            # Candidate = the step this trial would take, from the current moments.
            dirs = self._compute_directions(scalars, seed, commit=False)
            self._move(dirs, -1.0)
            loss_cand = closure().item()
            self._move(dirs, +1.0)  # restore θ
            if loss_cand < l_best:
                l_best, best = loss_cand, (j, scalars)

        # The best trial is applied only if it improves on the centre and passes the window.
        accepted = best is not None and self._kron_accepts(center, l_best)
        if accepted:
            j, scalars = best
            # Same state, same inputs → the same step that was just evaluated; now committed.
            dirs = self._compute_directions(scalars, self._trial_seed(z_seed, j), commit=True)
            self._move(dirs, -1.0)
            self._decay(dirs)
        else:
            self._last_metrics["grad_est_norm"] = 0.0
            self._last_metrics["subspace_grad_norm"] = 0.0
        self._push_window(min(center, l_best))

        self._n_steps += 1
        self._n_accepted += int(accepted)
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        return all_scalars


class SubZeroPlus(_KronDirectional, ZeroOrderOptimizer):
    """SubZero+: multi-query ZO in a Haar-random low-rank subspace with in-subspace Adam.

    Reference: "SubZero+: Efficient Zeroth-Order LLM Fine-Tuning via Large
    Learning Rates", arXiv:2608.15665.  Builds on SubZero (Yu et al., 2025).

    For each 2-D weight ``W ∈ R^{m×n}`` two column-orthonormal projections
    ``U ∈ R^{m×r}`` and ``V ∈ R^{n×r}`` are drawn by Haar-corrected QR and held
    fixed for ``step_interval`` (=F) steps.  Perturbations live in the induced
    ``r²``-dimensional subspace::

        Z̃⁽ᵏ⁾ = U Z⁽ᵏ⁾ Vᵀ,        Z⁽ᵏ⁾ ∈ R^{r×r},  Z⁽ᵏ⁾_ab ~ N(0, 1)

    Because ``vec(Z̃) = (V ⊗ U) vec(Z)`` and ``V ⊗ U`` is column-orthonormal,
    the map is an isometry and the estimator variance drops from ``O(m·n)`` to
    ``O(r²)`` per layer.

    Each step evaluates one shared baseline ``ℓ₀`` and ``K`` perturbed losses
    (``K+1`` forward passes, not ``2K``), all sharing the *same* ``U, V``::

        ρ_k = (ℓ_k − ℓ₀) / ε,       Z̄ = (1/K) Σ_k ρ_k Z⁽ᵏ⁾  ∈ R^{r×r}

    ``Z̄`` is the r×r coordinate representation of the gradient estimate; the
    full-space estimate ``U Z̄ Vᵀ`` is never materialized.  Adam then runs
    *entirely inside* that r×r space::

        M ← β₁M + (1−β₁)Z̄,   V ← β₂V + (1−β₂)(Z̄ ⊙ Z̄)
        W ← W − lr · U (M̂ / (√V̂ + ε_adam)) Vᵀ

    so the persistent optimizer state is ``2r²`` floats per layer instead of
    ``2·m·n`` — this is what makes multi-query Adam affordable at all.

    1-D parameters (biases, norm weights) have no subspace structure and fall
    back to plain MeZO-style probing, with Adam applied elementwise in full
    space (negligible memory: 1-D params are a tiny fraction of ``d``).

    DDP: ``U, V`` are seeded from ``_step_count``, which is globally
    synchronized, so every rank builds an identical subspace; the ``Z⁽ᵏ⁾`` come
    from the externally broadcast ``z_seed``.  The closure all-reduces the loss,
    so all ranks compute the same scalars and apply the same update without any
    gradient communication.  FSDP is not supported (blocked in
    ``build_optimizer``).

    Notes on deviations from the paper, all opt-in:

    * ``perturbation_mode="two_side"`` uses the central difference
      ``(ℓ₊−ℓ₋)/2ε`` at ``2K`` forwards per step.  The paper's forward
      difference is cheaper but carries an ``O(ε)`` bias that averaging over
      ``K`` does not remove (averaging kills variance, not bias).
    * ``moment_policy`` controls what happens to the Adam buffers when ``U, V``
      are redrawn.  The paper keeps them ("keep"), but after a refresh the r×r
      coordinates refer to entirely different directions in full space, so the
      carried-over statistics are stale.  See ``_refresh_projections``.
    * ``update_rule="sgd"`` drops Adam and steps along ``U Z̄ Vᵀ`` directly
      (this is SubZero + multi-query); the paper reports it as competitive.
    * ``kron_query_budget = q > 0`` adds KronZO's directional selection (see
      ``_KronDirectional``): q trials, each a full K-probe estimate + update rule, and the
      best candidate step is applied.  ``1+q(K+1)`` forwards (one_side) or ``1+q(2K+1)``
      (two_side, which then also needs a centre forward).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 24,
        step_interval: int = 50,
        num_queries: int = 8,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_rule: str = "adam",
        haar: bool = True,
        moment_policy: str = "keep",
        perturbation_mode: str = "one_side",
        kron_query_budget: int = 0,
        kron_history_length: int = 10,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if step_interval < 1:
            raise ValueError(f"Invalid step_interval: {step_interval}")
        if num_queries < 1:
            raise ValueError(f"Invalid num_queries: {num_queries}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if update_rule not in ("adam", "sgd"):
            raise ValueError("update_rule must be 'adam' or 'sgd'")
        if moment_policy not in ("keep", "zero", "project"):
            raise ValueError("moment_policy must be 'keep', 'zero' or 'project'")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            num_queries=num_queries,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            update_rule=update_rule,
            haar=haar,
            moment_policy=moment_policy,
            perturbation_mode=perturbation_mode,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}
        self._init_kron(kron_query_budget, kron_history_length)
        self._adam_steps = 0  # committed Adam updates; drives bias correction

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return SubZero+ diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs
            Mean ``|ρ_k|`` over the ``K`` queries — average signal strength
            along the probed subspace directions.
        projected_grad_std
            Standard deviation of ``ρ_k`` across the ``K`` queries.  This is the
            quantity multi-query estimation exists to average down; watch
            ``projected_grad_abs / projected_grad_std`` as an SNR proxy.
        subspace_grad_norm
            Global norm of the aggregated low-rank gradient ``Z̄`` across
            parameters.  Since ``U, V`` are column-orthonormal this *equals*
            ``‖U Z̄ Vᵀ‖_F``, i.e. the norm of the full-space gradient estimate,
            computed at r×r cost.
        grad_est_norm
            Global norm of the actual update direction before ``lr`` scaling —
            ``‖M̂/(√V̂+ε)‖_F`` under Adam, ``‖Z̄‖_F`` under SGD.  Under Adam this
            is roughly ``sqrt(#coords)`` because the ratio is ~unit-scale, so it
            tracks how much of the subspace is being used rather than gradient
            magnitude.
        grad_est_norm_per_z_rms
            ``grad_est_norm`` divided by the RMS norm of the raw probes, for
            comparability with the rest of the ZO family.
        update_accepted, accept_rate
            KronZO selection diagnostics; only reported with ``kron_query_budget > 0``.
            In that mode ``subspace_grad_norm`` / ``grad_est_norm`` describe the selected
            trial (0 on a rejected step) and ``projected_grad_*`` cover all ``q·K`` probes.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _effective_rank(self, p: torch.Tensor, group: dict) -> int:
        return min(group["rank"], p.shape[0], p.shape[1])

    def _get_or_refresh_projections(
        self, p: torch.Tensor, param_idx: int, group: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached ``(U, V)``, redrawing them every ``step_interval`` steps."""
        state = self.state[p]
        last_refresh = state.get("proj_refresh_step", -1)
        should_refresh = (last_refresh < 0) or (
            self._step_count - last_refresh >= group["step_interval"]
        )
        if should_refresh:
            r = self._effective_rank(p, group)
            base = self._step_count * 1_000_003 + param_idx * 999_983
            u_new = _haar_orthonormal(p.shape[0], r, p.device, base, group["haar"])
            v_new = _haar_orthonormal(p.shape[1], r, p.device, base + 7_919, group["haar"])
            if last_refresh >= 0:
                self._remap_moments(state, u_new, v_new, group["moment_policy"])
            state["u_mat"] = u_new
            state["v_mat"] = v_new
            state["proj_refresh_step"] = self._step_count
        return state["u_mat"], state["v_mat"]

    @staticmethod
    def _remap_moments(
        state: dict, u_new: torch.Tensor, v_new: torch.Tensor, policy: str
    ) -> None:
        """Handle the Adam buffers when the subspace basis is redrawn.

        The buffers are indexed by r×r *coordinates*, which are meaningless once
        ``U, V`` change: coordinate (a, b) named direction ``u_a v_bᵀ`` before
        the refresh and a different one after.

        * ``keep``    — do nothing (what the paper does).  Cheap, and harmless
                        when the run is short relative to ``step_interval``.
        * ``zero``    — restart Adam in the new basis.  Safe but loses the
                        per-layer step-size scale and re-triggers the bias
                        correction warm-up.
        * ``project`` — re-express the first moment in the new basis via the
                        change-of-basis ``M ← (Uₙᵀ Uₒ) M (Vₒᵀ Vₙ)``, which is
                        exact for the component of ``M`` that survives into the
                        new subspace and drops the rest.  The second moment is
                        *not* a linear object, so rotating it is not meaningful;
                        instead its mean is broadcast, preserving the per-layer
                        step-size scale while discarding stale per-coordinate
                        structure.
        """
        if policy == "keep" or "exp_avg" not in state:
            return
        if policy == "zero":
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            return
        u_old, v_old = state["u_mat"], state["v_mat"]
        if u_old.shape[1] != u_new.shape[1] or v_old.shape[1] != v_new.shape[1]:
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            return
        state["exp_avg"] = (u_new.T @ u_old) @ state["exp_avg"] @ (v_old.T @ v_new)
        state["exp_avg_sq"] = torch.full_like(
            state["exp_avg_sq"], state["exp_avg_sq"].mean().item()
        )

    def _get_z(
        self, p: torch.Tensor, z_seed: int, param_idx: int, query_idx: int, group: dict
    ) -> torch.Tensor:
        """Sample the in-subspace probe ``Z ∈ R^{r×r}`` (2-D) or ``z`` (1-D), float32."""
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + param_idx * 1_000_003 + query_idx * 999_983) % (2**31))
        shape = (
            (self._effective_rank(p, group),) * 2 if p.dim() == 2 else tuple(p.shape)
        )
        return torch.randn(shape, device=p.device, dtype=torch.float32, generator=gen)

    def _perturb(self, z_seed: int, query_idx: int, scaling_factor: float) -> None:
        """θ ← θ + scaling·ε·U Z Vᵀ (2-D) or θ + scaling·ε·z (1-D)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            eps = group["zo_eps"]
            z = self._get_z(p, z_seed, idx, query_idx, group)
            if p.dim() == 2:
                u, v = self._get_or_refresh_projections(p, idx, group)
                delta = (u @ z) @ v.T
                p.data.add_(delta.to(p.dtype), alpha=scaling_factor * eps)
            else:
                p.data.add_(z.to(p.dtype), alpha=scaling_factor * eps)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        zo_eps = self.defaults["zo_eps"]
        num_queries = self.defaults["num_queries"]
        two_side = self.defaults["perturbation_mode"] == "two_side"
        first_loss: Optional[torch.Tensor] = None

        if self._kron_q > 0 or not two_side:
            # A centre loss is needed for the shared one_side baseline and for KronZO's
            # candidate comparison. In plain two_side mode it is skipped (2K forwards).
            first_loss = closure()

        if self._kron_q > 0:
            scalars = self._kron_select_step(closure, first_loss, z_seed, num_queries, two_side, zo_eps)
        else:
            scalars, first_plus = self._estimate(closure, first_loss, z_seed, num_queries, two_side, zo_eps)
            if first_loss is None:
                first_loss = first_plus
            self._apply_update(scalars, z_seed)

        rho = list(scalars.values())
        mean_rho = sum(rho) / len(rho)
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for s in rho) / len(rho)
        self._last_metrics["projected_grad_std"] = (
            math.sqrt(sum((s - mean_rho) ** 2 for s in rho) / len(rho)) if len(rho) > 1 else 0.0
        )
        self._step_count += 1
        return first_loss  # type: ignore[return-value]

    def _estimate(
        self,
        closure,
        center_loss: Optional[torch.Tensor],
        z_seed: int,
        num_queries: int,
        two_side: bool,
        zo_eps: float,
    ) -> tuple[dict[int, float], Optional[torch.Tensor]]:
        """K probes → SPSA scalars. Returns them and the first ``L(θ+εΔ)`` (two_side only)."""
        scalars: dict[int, float] = {}
        first_plus: Optional[torch.Tensor] = None
        if two_side:
            # Central difference: 2K forwards, unbiased, no shared baseline.
            for k in range(num_queries):
                self._perturb(z_seed, k, +1.0)
                loss_plus = closure()
                if first_plus is None:
                    first_plus = loss_plus
                self._perturb(z_seed, k, -2.0)
                loss_minus = closure()
                self._perturb(z_seed, k, +1.0)  # restore θ
                scalars[k] = (loss_plus - loss_minus).item() / (2.0 * zo_eps)
        else:
            # Paper's scheme: one shared baseline + K perturbed → K+1 forwards.
            for k in range(num_queries):
                self._perturb(z_seed, k, +1.0)
                loss_k = closure()
                self._perturb(z_seed, k, -1.0)  # restore θ
                scalars[k] = (loss_k - center_loss).item() / zo_eps
        return scalars, first_plus

    def _apply_update(self, scalars: dict[int, float], z_seed: int) -> None:
        dirs = self._compute_directions(scalars, z_seed, commit=True)
        self._move(dirs, -1.0)
        self._decay(dirs)

    def _compute_directions(self, scalars: dict[int, float], z_seed: int, commit: bool) -> list:
        """Aggregate ``Z̄`` and return the coefficient-space step directions
        ``[(group, p, coeff), ...]`` (before ``lr``); nothing is applied.  With
        ``commit=True`` the Adam state advances; with ``commit=False`` it is left untouched
        and the direction is what a committed call would produce from the same state."""
        num_queries = len(scalars)
        t = self._adam_steps + 1  # Adam bias-correction step, 1-based
        grad_sum_sq = 0.0
        subspace_sum_sq = 0.0
        z_sum_sq = 0.0
        dirs = []

        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            # Z̄ = (1/K) Σ_k ρ_k Z⁽ᵏ⁾ — the r×r coordinates of the estimate.
            z_bar = torch.zeros_like(
                self._get_z(p, z_seed, idx, 0, group), dtype=torch.float32
            )
            for query_idx, rho in scalars.items():
                z_k = self._get_z(p, z_seed, idx, query_idx, group)
                z_bar.add_(z_k, alpha=rho)
                z_sum_sq += z_k.norm().item() ** 2
            z_bar.div_(num_queries)
            # U, V column-orthonormal ⇒ ‖U Z̄ Vᵀ‖_F = ‖Z̄‖_F, so this is the
            # full-space gradient-estimate norm at r×r cost.
            subspace_sum_sq += z_bar.norm().item() ** 2

            if group["update_rule"] == "adam":
                state = self.state[p]
                exp_avg, exp_avg_sq = state.get("exp_avg"), state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg, exp_avg_sq = torch.zeros_like(z_bar), torch.zeros_like(z_bar)
                    if commit:
                        state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
                if commit:
                    exp_avg.mul_(beta1).add_(z_bar, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(z_bar, z_bar, value=1.0 - beta2)
                else:
                    exp_avg = exp_avg.mul(beta1).add(z_bar, alpha=1.0 - beta1)
                    exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(z_bar, z_bar, value=1.0 - beta2)
                m_hat = exp_avg / (1.0 - beta1**t)
                v_hat = exp_avg_sq / (1.0 - beta2**t)
                direction = m_hat / (v_hat.sqrt() + adam_eps)
            else:
                direction = z_bar

            grad_sum_sq += direction.norm().item() ** 2
            dirs.append((group, p, direction))

        if commit:
            self._adam_steps += 1
        ge = math.sqrt(max(grad_sum_sq, 0.0))
        z_rms = math.sqrt(z_sum_sq) if z_sum_sq > 0.0 else 0.0
        self._last_metrics["subspace_grad_norm"] = math.sqrt(max(subspace_sum_sq, 0.0))
        self._last_metrics["grad_est_norm"] = ge
        self._last_metrics["grad_est_norm_per_z_rms"] = (ge / z_rms) if z_rms > 0.0 else float("nan")
        return dirs

    # ------------------------------------------------------------------
    # Moves (shared with the KronZO selection)
    # ------------------------------------------------------------------

    def _lift(self, p: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """Coefficients → full space: ``U c Vᵀ`` for 2-D weights, identity otherwise."""
        if p.dim() == 2:
            state = self.state[p]
            return (state["u_mat"] @ coeff) @ state["v_mat"].T
        return coeff

    def _move(self, dirs: list, sign: float) -> None:
        """θ += sign · lr · lift(coeff)."""
        for group, p, coeff in dirs:
            p.data.add_(self._lift(p, coeff).to(p.dtype), alpha=sign * group["lr"])

    def _decay(self, dirs: list) -> None:
        for group, p, _coeff in dirs:
            if group["weight_decay"] != 0.0:
                p.data.mul_(1.0 - group["lr"] * group["weight_decay"])


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization  (Muon / ZO-Muon building block)
# ---------------------------------------------------------------------------

def _newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate matrix sign via Newton-Schulz iteration.

    Maps G → U Vᵀ where G = U Σ Vᵀ, i.e. normalizes all singular values to ~1.
    This is the "Muon" update rule from Kosson et al. and Vyas et al.
    Always operates in float32 and returns float32.

    Convergence requires spectral norm < sqrt(3); the initial normalisation
    by ``X.norm()`` ensures this.
    """
    assert G.ndim == 2, "G must be a 2-D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ---------------------------------------------------------------------------
# ZOMuon
# ---------------------------------------------------------------------------

class ZOMuon(ZeroOrderOptimizer):
    """Zeroth-Order optimizer with subspace gradient orthogonalization (ZO-Muon).

    For 2-D weight matrices the algorithm:
      1. Samples a Haar-distributed orthogonal projection ``P`` of shape
         ``(m, rank)`` via QR; cached and refreshed every ``step_interval`` steps.
      2. Accumulates ``num_samples`` gradient scalars in the low-rank subspace:
           ``lowdim_rge = (1/N) Σᵢ scalar_i · uᵢ``   shape ``(rank, n)``
         where each ``uᵢ ~ N(0,I)`` is the per-sample in-subspace direction.
         With ``num_samples=1`` the antithetic two-sided estimator is used
         (2 forward passes); with ``num_samples>1`` a one-sided centre-based
         estimator is used (1+N forward passes, shared ``P``).
      3. Applies Newton-Schulz orthogonalization:
           ``M_sign = NS(lowdim_rge)``   (≈ U Vᵀ, singular values → 1)
      4. Updates: ``W -= lr · P @ M_sign``

    For 1-D parameters (biases, norms) a standard averaged MeZO-style update
    is used without Newton-Schulz.

    DDP: ``P`` is seeded from ``_step_count`` (globally synchronized across
    ranks); ``uᵢ`` from the externally broadcast ``z_seed``.  The loss is
    all-reduced inside the closure so all ranks compute identical scalars and
    apply the same update without gradient communication.

    FSDP is not supported (blocked in ``build_optimizer``).

    Reference: "Powering Up Zeroth-Order Training via Subspace Gradient
    Orthogonalization", 2024.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 4,
        step_interval: int = 1,
        num_samples: int = 1,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if num_samples < 1:
            raise ValueError(f"Invalid num_samples: {num_samples}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            step_interval=step_interval,
            num_samples=num_samples,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return ZOMuon diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs_mean
            Mean absolute SPSA scalar across all ``num_samples`` samples.
            Measures the average signal strength in the probed subspace.
        grad_est_norm
            Global L2 / Frobenius norm of the update direction
            ``P @ M_sign`` for 2-D params (plus standard norm for 1-D),
            before ``lr`` scaling.  Because Newton-Schulz normalizes singular
            values to ~1, this is roughly ``sqrt(rank · n_2d_params)``.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_or_refresh_p(self, p: torch.Tensor, param_idx: int) -> torch.Tensor:
        """Return the cached Haar-orthogonal P, refreshing if the interval has elapsed."""
        effective_rank = min(self.defaults["rank"], p.shape[0])
        state = self.state[p]
        last_refresh = state.get("p_refresh_step", -1)
        should_refresh = (last_refresh < 0) or (
            self._step_count - last_refresh >= self.defaults["step_interval"]
        )
        if should_refresh:
            gen = torch.Generator(device=p.device)
            gen.manual_seed(
                (self._step_count * 1_000_003 + param_idx * 999_983) % (2**31)
            )
            Q = torch.randn(
                p.shape[0], effective_rank, device=p.device, dtype=torch.float32, generator=gen
            )
            P, R = torch.linalg.qr(Q, mode="reduced")
            # Haar sign correction: ensure uniform distribution over O(m, rank).
            signs = R.diagonal().sign()
            signs[signs == 0] = 1.0
            P = P * signs  # float32, shape (m, effective_rank)
            state["p_mat"] = P
            state["p_refresh_step"] = self._step_count
        return state["p_mat"]

    def _get_u(
        self, p: torch.Tensor, z_seed: int, param_idx: int, sample_idx: int
    ) -> torch.Tensor:
        """Sample uᵢ ~ N(0, I) of shape (rank, n_cols) for sample i. Always float32."""
        effective_rank = min(self.defaults["rank"], p.shape[0])
        gen = torch.Generator(device=p.device)
        gen.manual_seed(
            (z_seed + param_idx * 1_000_003 + sample_idx * 999_983) % (2**31)
        )
        return torch.randn(
            effective_rank, p.shape[1], device=p.device, dtype=torch.float32, generator=gen
        )

    def _get_z1d(
        self, p: torch.Tensor, z_seed: int, param_idx: int, sample_idx: int
    ) -> torch.Tensor:
        """Sample z ~ N(0, I) for 1-D params. Always float32."""
        gen = torch.Generator(device=p.device)
        gen.manual_seed(
            (z_seed + param_idx * 999_983 + sample_idx * 111_317) % (2**31)
        )
        return torch.randn(p.shape, device=p.device, dtype=torch.float32, generator=gen)

    def _perturb_sample(self, z_seed: int, sample_idx: int, scaling_factor: float) -> None:
        """Perturb all parameters for one sample: θ += scaling * eps * P @ u (2D) or z (1D)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            eps = group["zo_eps"]
            if p.dim() == 2:
                P = self._get_or_refresh_p(p, idx)
                u = self._get_u(p, z_seed, idx, sample_idx)
                p.data.addmm_(P.to(p.dtype), u.to(p.dtype), alpha=scaling_factor * eps)
            else:
                z = self._get_z1d(p, z_seed, idx, sample_idx)
                p.data.add_(z.to(p.dtype), alpha=scaling_factor * eps)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        zo_eps = self.defaults["zo_eps"]
        num_samples = self.defaults["num_samples"]
        scalars: dict[int, float] = {}

        if num_samples == 1:
            # Two-sided antithetic: 2 forward passes, lower-variance estimator.
            self._perturb_sample(z_seed, 0, +1.0)
            loss_plus = closure()
            self._perturb_sample(z_seed, 0, -2.0)
            loss_minus = closure()
            scalars[0] = (loss_plus - loss_minus).item() / (2.0 * zo_eps)
            self._perturb_sample(z_seed, 0, +1.0)  # restore θ
        else:
            # One-sided centre-based: 1 + num_samples forward passes, shared P.
            loss_center = closure()
            loss_plus = loss_center  # fallback; overwritten below
            for i in range(num_samples):
                self._perturb_sample(z_seed, i, +1.0)
                loss_plus_i = closure()
                if i == 0:
                    loss_plus = loss_plus_i  # return first sample's loss for logging
                self._perturb_sample(z_seed, i, -1.0)  # restore θ
                scalars[i] = (loss_plus_i - loss_center).item() / zo_eps

        self._apply_update(scalars, z_seed)
        self._last_metrics["projected_grad_abs_mean"] = (
            sum(abs(s) for s in scalars.values()) / len(scalars)
        )
        self._step_count += 1
        return loss_plus

    def _apply_update(self, scalars: dict[int, float], z_seed: int) -> None:
        ns_steps = self.defaults["ns_steps"]
        n_samples = len(scalars)
        grad_sum_sq = 0.0

        for group, p, idx in self._flat_params():
            if not p.requires_grad:
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            if p.dim() == 2:
                P = self.state[p]["p_mat"]  # (m, effective_rank), float32
                effective_rank = P.shape[1]

                # Accumulate low-rank gradient estimate: (1/N) Σᵢ scalarᵢ · uᵢ
                lowdim_rge = torch.zeros(
                    effective_rank, p.shape[1], device=p.device, dtype=torch.float32
                )
                for sample_idx, scalar in scalars.items():
                    u = self._get_u(p, z_seed, idx, sample_idx)
                    lowdim_rge.add_(u, alpha=scalar)
                lowdim_rge.div_(n_samples)

                # Newton-Schulz: lowdim_rge → M_sign ≈ U Vᵀ  (singular values → 1)
                M_sign = _newtonschulz5(lowdim_rge, steps=ns_steps)

                # Full update direction G = P @ M_sign, shape (m, n)
                G = P @ M_sign
                grad_sum_sq += G.norm().item() ** 2

                p.data.add_(G.to(p.dtype), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

            else:
                # 1-D: average the scalar gradient estimates, no orthogonalization.
                grad_1d = torch.zeros(p.shape, device=p.device, dtype=torch.float32)
                for sample_idx, scalar in scalars.items():
                    z = self._get_z1d(p, z_seed, idx, sample_idx)
                    grad_1d.add_(z, alpha=scalar)
                grad_1d.div_(n_samples)

                grad_sum_sq += grad_1d.norm().item() ** 2
                p.data.add_(grad_1d.to(p.dtype), alpha=-lr)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

        self._last_metrics["grad_est_norm"] = math.sqrt(grad_sum_sq)


def _best_pair(n: int) -> tuple[int, int]:
    """Two divisors of ``n`` closest to ``sqrt(n)`` (Algorithm 1 ``BestPair`` in KronZO).

    Returns ``(1, n)`` for primes (the paper notes this is rare for LM layer widths).
    """
    if n <= 1:
        return (1, n)
    root = int(math.isqrt(n))
    for offset in range(root + 1):
        for cand in (root - offset, root + offset):
            if cand >= 1 and n % cand == 0:
                return (cand, n // cand)
    return (1, n)  # fallback for primes


# ---------------------------------------------------------------------------
# Diagonal-Hessian preconditioner (HiZOO) — shared mix-in
# ---------------------------------------------------------------------------

class _HessianPreconditioner:
    """Diagonal-Hessian pre-conditioner (HiZOO, ICLR 2025) as a reusable mix-in.

    Grafts HiZOO's curvature scaling onto any two-sided ZO optimizer that samples a
    probe direction ``z`` per query. Per parameter we keep a diagonal Hessian estimate
    ``Σ⁻¹`` (init ``1``) and expose ``Σ^½ = 1/√Σ⁻¹`` as an elementwise pre-conditioner:
    every perturbation / step is taken along ``Σ^½ ⊙ z`` instead of ``z`` (large steps
    where the loss is flat, small where it is sharp).

    Once per step the host feeds back the finite-difference curvature along each probe,
    ``S_i = (ℓ₊ + ℓ₋ − 2ℓ) / (2ε²) ≈ zᵢᵀ H zᵢ`` (HiZOO Eq. 3-4 / Algorithm 1, line 9),
    and the Hessian is EMA-updated from the raw ``z`` (not the pre-conditioned one):

        Σ' = Σ⁻¹ · (1/q) Σᵢ Sᵢ zᵢ² ,   Σ⁻¹ ← (1−α) Σ⁻¹ + α |Σ'| ,   clamp(min=hess_min).

    In two-sided mode ``ℓ, ℓ₊, ℓ₋`` are already evaluated, so the curvature costs **no
    extra forward pass**; the only overhead is one ``O(d)`` float32 buffer per parameter
    (like storing a gradient — still far below first-order, which also holds activations).

    The old ``Σ`` is used for the whole step (probe, candidate and applied update) and the
    EMA update runs at the very end, so the applied step matches the candidate that was
    selected; the one-step lag is negligible for the tiny EMA rate ``α``.
    """

    def _init_hessian(self, enabled: bool, smooth: float, hess_min: float) -> None:
        if enabled and not (0.0 <= smooth <= 1.0):
            raise ValueError(f"hessian_smooth (EMA α) must be in [0, 1], got {smooth}")
        if enabled and hess_min <= 0:
            raise ValueError(f"hessian_min must be > 0, got {hess_min}")
        self._hessian_on = bool(enabled)
        self._hessian_smooth = smooth
        self._hessian_min = hess_min

    def _hess(self, p: torch.Tensor) -> torch.Tensor:
        """Per-parameter diagonal Hessian estimate ``Σ⁻¹`` (float32), lazily set to ones."""
        state = self.state[p]  # type: ignore[attr-defined]
        if "hess" not in state:
            state["hess"] = torch.ones_like(p, dtype=torch.float32)
        return state["hess"]

    def _precond(self, p: torch.Tensor) -> Optional[torch.Tensor]:
        """``Σ^½ = 1/√Σ⁻¹`` (float32), or ``None`` when the pre-conditioner is off."""
        if not getattr(self, "_hessian_on", False):
            return None
        return torch.rsqrt(self._hess(p))

    def _apply_precond(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return ``Σ^½ ⊙ z`` (in ``p``'s dtype) when enabled, else ``z`` unchanged."""
        pc = self._precond(p)
        if pc is None:
            return z
        return (z.float() * pc).to(p.dtype)

    @staticmethod
    def _curvature(l_center: float, l_plus: float, l_minus: float, eps: float) -> float:
        """Finite-difference curvature ``S = (ℓ₊ + ℓ₋ − 2ℓ) / (2ε²)`` along the probe."""
        return (l_plus + l_minus - 2.0 * l_center) / (2.0 * eps * eps)

    def _hessian_ema(self, curv_z2: dict[int, torch.Tensor], q: int) -> None:
        """One EMA update of ``Σ⁻¹`` from the accumulated ``Σᵢ Sᵢ zᵢ²`` (+ diagnostics)."""
        alpha = self._hessian_smooth
        hess_min = self._hessian_min
        hess_sum = precond_sum = 0.0
        n_elems = 0
        for group in self.param_groups:  # type: ignore[attr-defined]
            for p in group["params"]:
                if not p.requires_grad or id(p) not in curv_z2:
                    continue
                hess = self._hess(p)
                sigma_prime = hess.mul(curv_z2[id(p)]).div_(q)  # Σ⁻¹ · (1/q) Σ Sᵢ zᵢ²
                hess.mul_(1.0 - alpha).add_(sigma_prime.abs_(), alpha=alpha)
                hess.clamp_(min=hess_min)
                hess_sum += hess.sum().item()
                precond_sum += torch.rsqrt(hess).sum().item()
                n_elems += hess.numel()
        if n_elems > 0:
            self._last_metrics["hessian_mean"] = hess_sum / n_elems  # type: ignore[attr-defined]
            self._last_metrics["precond_sqrt_mean"] = precond_sum / n_elems  # type: ignore[attr-defined]


class KronZO(_HessianPreconditioner, ZeroOrderOptimizer):
    """Zeroth-order Kronecker optimization for LM pretraining (KronZO).

    Implements "Zeroth-order Kronecker optimization for pretraining language models"
    (Allaire, Le Digabel, Orban, Partovi Nia, GERAD G-2025-44).

    Two ideas vs. MeZO / LOZO:

    1. **Kronecker-factored perturbations** (Algorithm 1).  For a 2-D weight
       ``W ∈ R^{m×n}`` the probe direction is ``Z = A ⊗ B`` where
       ``A ∈ R^{m1×n1}``, ``B ∈ R^{m2×n2}`` with ``m1·m2 = m`` and ``n1·n2 = n``,
       each factor chosen as close to square as possible.  Because
       ``rank(A⊗B) = rank(A)·rank(B)``, ``Z`` is almost surely **full rank** (so it
       explores the whole subspace, unlike low-rank LOZO) while only the tiny
       factors are stored.  ``B`` is frozen for ``step_interval`` steps (like ``V``
       in LOZO); ``A`` is resampled for every probe.  1-D params use a plain
       Gaussian probe.

    2. **Criterion-driven directional update** (Algorithm 2, "lucky RGE").  Each
       step samples ``query_budget`` (``q``) probes; for each it forms the SPSA
       scalar ``c_i = (L(θ+εZ_i) − L(θ−εZ_i)) / 2ε`` and *evaluates the candidate
       step* ``L(θ − α c_i Z_i)``.  Only the single best-decreasing direction is
       kept (not the average).  The step is then accepted only if that best loss
       beats the worst of the last ``history_length`` (``h``) accepted losses;
       otherwise the step is skipped.  Cost per step: ``1 + 3q`` forward passes
       (two-sided) or ``1 + 2q`` (one-sided).

    Probes are regenerated from seeds rather than stored, keeping the live memory
    footprint at one parameter-sized temporary.  DDP: ``A`` is seeded from the
    externally broadcast ``z_seed`` and ``B`` from the globally-synchronized
    ``_step_count``, while the closure all-reduces the loss, so every rank derives
    identical scalars, accept/reject decisions and updates without gradient
    communication.  FSDP is not supported (blocked in ``build_optimizer``).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        step_interval: int = 50,
        query_budget: int = 10,
        history_length: int = 10,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        orthogonal_probes: bool = False,
        momentum: float = 0.0,
        line_search_steps: int = 0,
        hessian: bool = False,
        hessian_smooth: float = 1e-6,
        hessian_min: float = 1e-8,
        layer_sparsity: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if step_interval < 1:
            raise ValueError(f"Invalid step_interval: {step_interval}")
        if query_budget < 1:
            raise ValueError(f"Invalid query_budget: {query_budget}")
        if history_length < 0:
            raise ValueError(f"Invalid history_length: {history_length}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1): {momentum}")
        if line_search_steps < 0:
            raise ValueError(f"line_search_steps must be >= 0: {line_search_steps}")
        if not (0.0 <= layer_sparsity < 1.0):
            raise ValueError(f"layer_sparsity must be in [0, 1): {layer_sparsity}")
        if hessian:
            # HiZOO curvature needs ℓ, ℓ₊, ℓ₋; the momentum/line-search paths don't
            # thread the pre-conditioner, so keep the combination out for now.
            if perturbation_mode != "two_side":
                raise ValueError("hessian preconditioner requires perturbation_mode='two_side'")
            if momentum != 0.0 or line_search_steps != 0:
                raise ValueError("hessian preconditioner is not supported with momentum / line_search")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            step_interval=step_interval,
            query_budget=query_budget,
            history_length=history_length,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            orthogonal_probes=orthogonal_probes,
            momentum=momentum,
            line_search_steps=line_search_steps,
            layer_sparsity=layer_sparsity,
        )
        super().__init__(params, defaults)
        self._init_hessian(hessian, hessian_smooth, hessian_min)
        self._step_count = 0
        # Sliding window of the last `history_length` accepted losses; prefilled with
        # +inf so the first `h` steps always pass the acceptance test.
        self._window: list[float] = [float("inf")] * history_length
        self._n_steps = 0
        self._n_accepted = 0
        self._last_metrics: dict[str, float] = {}
        # Per-step probe bookkeeping for the optional ZO-vs-FO cosine diagnostic:
        # the (seed, scalar) of every probe and the index of the lowest-loss one.
        self._probe_samples: list[tuple[int, float]] = []
        self._best_probe_idx: int = -1
        # Orthogonal-probe support: seeds of the current step's q probes and a per-step,
        # per-parameter cache of the QR-orthogonalized A factors (rebuilt each step).
        self._probe_seeds: list[int] = []
        self._seed_pos: dict[int, int] = {}
        self._ortho_cache: dict[int, torch.Tensor] = {}
        # LeZO layer sparsity: the set of param indices active (perturbed + updated) this
        # step. Empty means "not yet selected"; _select_active fills it each step. When
        # layer_sparsity == 0 this is always the full set (plain KronZO).
        self._active_idx: set[int] = set()
        self._active_frac: float = 1.0

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return KronZO diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs
            Mean ``|c_i|`` over the ``q`` probes (SPSA signal strength).
        grad_est_norm
            Frobenius norm of the applied update direction ``c_best · Z_best``
            (0.0 when the step was skipped).  Computed cheaply via the Kronecker
            identity ``||A⊗B||_F = ||A||_F · ||B||_F`` without materializing ``Z``.
        update_accepted
            1.0 if the directional update was applied this step, else 0.0
            (a step is skipped when no probe improved on the center loss, or the
            best loss did not beat the acceptance window).
        accept_rate
            Running fraction of accepted steps (the paper observes ~90%).
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Cosine similarity between KronZO's ZO gradient estimate and the true FO gradient.

        Must be called right after ``step()`` and after a real backward has populated
        ``p.grad`` for these parameters at the *same* weights θ_k (the trainer arranges
        this on its probe interval).  Reconstructs the probe directions from the seeds
        stored during the last step and reports two alignments:

        ``cosine_selected``
            cos(∠) between the *lucky* probe KronZO actually uses (lowest candidate
            loss, ``c_best·Z_best``) and ∇L.  How good the chosen step direction is.
        ``cosine_qrge``
            cos(∠) between the unbiased q-RGE average ``(1/q)·Σ c_i Z_i`` and ∇L.
            How good the raw averaged estimator is (improves with q, degrades with
            dimension — the "dilution" the directional update is designed to beat).

        Returns ``{}`` if no probe data or no gradients are available.
        """
        samples = self._probe_samples
        if not samples or self._best_probe_idx < 0:
            return {}
        q = len(samples)
        best_seed, best_c = samples[self._best_probe_idx]

        dot_sel = norm_sel = 0.0
        dot_avg = norm_avg = 0.0
        norm_g = 0.0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or p.grad is None or not self._active(idx):
                continue
            g = p.grad.float()
            norm_g += g.pow(2).sum().item()

            # Selected ("lucky") direction c_best · Z_best.
            sel = self._build_z(p, idx, best_seed).float().mul_(best_c)
            dot_sel += (g * sel).sum().item()
            norm_sel += sel.pow(2).sum().item()

            # Unbiased q-RGE average (1/q) Σ c_i Z_i.
            avg = torch.zeros_like(g)
            for seed_i, c_i in samples:
                avg.add_(self._build_z(p, idx, seed_i).float(), alpha=c_i)
            avg.div_(q)
            dot_avg += (g * avg).sum().item()
            norm_avg += avg.pow(2).sum().item()

        if norm_g == 0.0:
            return {}
        eps = 1e-12
        return {
            "cosine_selected": dot_sel / (math.sqrt(norm_g * norm_sel) + eps),
            "cosine_qrge": dot_avg / (math.sqrt(norm_g * norm_avg) + eps),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _select_active(self) -> None:
        """LeZO layer sparsity: choose which parameter tensors this step perturbs and updates.

        The active subset rotates deterministically with ``_step_count`` (a torch.randperm
        seeded from the step count on CPU), so it is identical on every DDP rank with no
        communication — the same design the frozen factor ``B`` already relies on. Over many
        steps every tensor is selected, so training remains full-parameter; each step just
        touches a ``(1 - layer_sparsity)`` fraction, cutting perturbation/update overhead and
        shrinking the probe dimension. ``layer_sparsity == 0`` keeps every tensor active
        (plain KronZO).
        """
        idxs = [idx for _g, p, idx in self._flat_params() if p.requires_grad]
        n_total = len(idxs)
        sparsity = self.defaults["layer_sparsity"]
        if sparsity <= 0.0 or n_total == 0:
            self._active_idx = set(idxs)
            self._active_frac = 1.0
            return
        n_keep = max(1, round(n_total * (1.0 - sparsity)))
        gen = torch.Generator()
        gen.manual_seed((self._step_count * 2_654_435_761 + 12_345) % (2**31))
        perm = torch.randperm(n_total, generator=gen).tolist()
        self._active_idx = {idxs[perm[i]] for i in range(n_keep)}
        self._active_frac = n_keep / n_total

    def _active(self, idx: int) -> bool:
        return idx in self._active_idx

    def _get_a(self, p: torch.Tensor, dims: tuple, z_seed: int, idx: int) -> torch.Tensor:
        m1, n1, _m2, _n2 = dims
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + idx * 1_000_003) % (2**31))
        return torch.randn(m1, n1, device=p.device, dtype=p.dtype, generator=gen)

    def _a_for(self, p: torch.Tensor, dims: tuple, seed: int, idx: int) -> torch.Tensor:
        """Return the A factor for one probe — raw Gaussian, or (if enabled) the
        QR-orthogonalized variant so the step's q probes are mutually orthogonal.

        Since ``<A_i⊗B, A_j⊗B> = <A_i,A_j>·||B||²`` (B is frozen within the step),
        orthogonalizing the small A factors makes the full perturbations Z_i orthogonal,
        maximizing coverage of the B-induced subspace per step. The orthonormal columns
        are rescaled to the Gaussian norm √(m1·n1) so ε / lr keep their scale. The whole
        set is reproducible from the step's seeds, so every regeneration point agrees.
        """
        if not self.defaults.get("orthogonal_probes", False):
            return self._get_a(p, dims, seed, idx)
        m1, n1, _m2, _n2 = dims
        q = len(self._probe_seeds)
        # Can't orthogonalize more vectors than the subspace dimension — fall back.
        if q < 2 or m1 * n1 < q:
            return self._get_a(p, dims, seed, idx)
        cache = self._ortho_cache.get(idx)
        if cache is None:
            raws = torch.stack(
                [self._get_a(p, dims, s, idx).float().reshape(-1) for s in self._probe_seeds],
                dim=1,
            )  # (m1*n1, q)
            qmat, _ = torch.linalg.qr(raws, mode="reduced")  # orthonormal columns (m1*n1, q)
            cache = (qmat * math.sqrt(m1 * n1)).t().reshape(q, m1, n1).to(p.dtype)  # (q, m1, n1)
            self._ortho_cache[idx] = cache
        return cache[self._seed_pos[seed]]

    def _refresh_b(self, p: torch.Tensor, dims: tuple, idx: int) -> torch.Tensor:
        _m1, _n1, m2, n2 = dims
        gen = torch.Generator(device=p.device)
        gen.manual_seed((self._step_count * 1_000_003 + idx * 999_983) % (2**31))
        return torch.randn(m2, n2, device=p.device, dtype=p.dtype, generator=gen)

    def _get_z1d(self, p: torch.Tensor, z_seed: int, idx: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed((z_seed + idx * 999_983) % (2**31))
        return torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)

    def _ensure_state(self) -> None:
        """Cache per-param Kronecker dims and (re)sample the frozen factor ``B``."""
        refresh = self._step_count % self.defaults["step_interval"] == 0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            if p.dim() != 2:
                continue
            state = self.state[p]
            if "kron_dims" not in state:
                m, n = p.shape
                m1, m2 = _best_pair(int(m))
                n1, n2 = _best_pair(int(n))
                state["kron_dims"] = (m1, n1, m2, n2)
            if refresh or "B" not in state:
                state["B"] = self._refresh_b(p, state["kron_dims"], idx)

    def _build_z(self, p: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
        """Regenerate the full perturbation ``Z`` for one parameter from ``seed``."""
        if p.dim() == 2:
            dims = self.state[p]["kron_dims"]
            a = self._a_for(p, dims, seed, idx)
            b = self.state[p]["B"]
            return torch.kron(a, b)
        return self._get_z1d(p, seed, idx)

    def _perturb_eps(self, seed: int, sign: float) -> None:
        """θ += sign · ε · (Σ^½ ⊙ Z)   (finite-difference probe, single global ε)."""
        eps = self.defaults["zo_eps"]
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            p.data.add_(self._apply_precond(p, self._build_z(p, idx, seed)), alpha=sign * eps)

    def _perturb_step(self, seed: int, c: float, sign: float) -> None:
        """θ += sign · lr · c · (Σ^½ ⊙ Z)   (candidate / trial step, per-group lr)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            p.data.add_(
                self._apply_precond(p, self._build_z(p, idx, seed)),
                alpha=sign * group["lr"] * c,
            )

    def _apply_best(self, seed: int, c: float) -> float:
        """Apply the accepted update ``θ -= lr · c · (Σ^½ ⊙ Z)`` (+ decoupled weight decay).

        Returns the Frobenius norm of the applied direction summed over params. Without
        the pre-conditioner this uses the Kronecker identity ``||A⊗B||_F = ||A||_F·||B||_F``;
        with it, ``Σ^½ ⊙ Z`` is materialized (as it must be anyway) and normed directly.
        """
        grad_sq = 0.0
        precond_on = getattr(self, "_hessian_on", False)
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            if p.dim() == 2 and not precond_on:
                dims = self.state[p]["kron_dims"]
                a = self._a_for(p, dims, seed, idx)
                b = self.state[p]["B"]
                grad_sq += (c * a.float().norm().item() * b.float().norm().item()) ** 2
                z = torch.kron(a, b)
            else:
                z = self._apply_precond(p, self._build_z(p, idx, seed))
                grad_sq += (c * z.float().norm().item()) ** 2
            p.data.add_(z, alpha=-lr * c)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
        return math.sqrt(max(grad_sq, 0.0))

    def _momentum_step(self, seed: Optional[int], c: Optional[float], accepted: bool) -> float:
        """Heavy-ball momentum update.

        Keeps a per-parameter buffer ``m`` and integrates the step's chosen ZO direction
        into it: ``m ← β·m + c_best·Z_best`` when the step is accepted (otherwise ``m`` just
        decays, ``m ← β·m``). The parameters always move along the accumulated direction,
        ``θ ← θ − lr·m`` (+ decoupled weight decay).

        Because the per-step estimates are (near-)unbiased, this averages many probes over
        time, so the effective alignment with the true gradient grows roughly like
        ``√(1/(1−β) / d)`` instead of ``1/√d`` for a single step — the cheapest way to
        "guess" the gradient better without any backward pass. Costs one extra full-size
        buffer per parameter (still far below first-order, which also stores activations).
        """
        beta = self.defaults["momentum"]
        upd_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            state = self.state[p]
            mom = state.get("mom")
            if mom is None:
                mom = torch.zeros_like(p, dtype=torch.float32)
                state["mom"] = mom
            if accepted:
                assert seed is not None and c is not None
                # EMA so ||m|| stays on the scale of a single estimate -> lr keeps its meaning
                # (heavy-ball would amplify the step by ~1/(1-β) and needs a smaller lr).
                mom.mul_(beta).add_(self._build_z(p, idx, seed).float(), alpha=c * (1.0 - beta))
            else:
                mom.mul_(beta)
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            p.data.add_(mom.to(p.dtype), alpha=-lr)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
            upd_sq += mom.norm().item() ** 2
        return math.sqrt(max(upd_sq, 0.0))

    def _step_along_momentum(self, sign: float) -> None:
        """θ += sign · lr · m   (move one more step along the momentum buffer)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(idx):
                continue
            mom = self.state[p].get("mom")
            if mom is None:
                continue
            p.data.add_(mom.to(p.dtype), alpha=sign * group["lr"])

    def _line_search(self, extend: Callable[[], None], undo: Callable[[], None],
                     closure: Callable[[], torch.Tensor], k_max: int) -> int:
        """Greedily keep stepping along the just-applied direction while the loss drops.

        Once a direction has been accepted, this rides it: re-apply the same update and
        evaluate the (same-batch) loss; keep it while it strictly improves, otherwise undo
        the last extension and stop. Amortizes the expensive 1+3q probe sampling over
        several real moves and acts as an adaptive step length. ``extend``/``undo`` move
        one step forward/back along the direction. Returns how many extra steps were taken.
        """
        cur = closure().item()
        taken = 0
        for _ in range(k_max):
            extend()
            nxt = closure().item()
            if nxt < cur:
                cur = nxt
                taken += 1
            else:
                undo()  # revert the non-improving extension
                break
        return taken

    def _push_window(self, loss: float) -> None:
        """Replace the worst (max) loss in the acceptance window with ``loss``."""
        if not self._window:
            return
        j = max(range(len(self._window)), key=lambda k: self._window[k])
        self._window[j] = loss

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        # LeZO: pick this step's active layer subset before anything touches the params
        # (_ensure_state only refreshes B for active tensors).
        self._select_active()
        self._ensure_state()
        eps = self.defaults["zo_eps"]
        q = self.defaults["query_budget"]
        two_side = self.defaults["perturbation_mode"] == "two_side"

        # Fix this step's q probe seeds up front so every regeneration point agrees,
        # and reset the per-step orthogonalization cache (depends on these seeds).
        self._probe_seeds = [(z_seed + i * 987_654_321) % (2**31) for i in range(q)]
        self._seed_pos = {s: i for i, s in enumerate(self._probe_seeds)}
        self._ortho_cache = {}

        # Line 5: center loss at θ_k. Also the value reported to the trainer.
        center_loss = closure()
        center = center_loss.item()
        l_best = center
        best_seed: Optional[int] = None
        best_c: Optional[float] = None
        abs_c_sum = 0.0
        samples: list[tuple[int, float]] = []
        best_probe_idx = -1
        best_probe_loss = float("inf")

        # HiZOO curvature: accumulate Σᵢ Sᵢ zᵢ² (raw z) per param for the Hessian EMA.
        hess_on = getattr(self, "_hessian_on", False)
        curv_abs_sum = 0.0
        curv_z2: dict[int, torch.Tensor] = (
            # Only active params get an entry: _hessian_ema skips ids absent from the dict,
            # so a LeZO-frozen tensor keeps its Hessian instead of decaying it toward zero.
            {id(p): torch.zeros_like(p, dtype=torch.float32)
             for _g, p, i in self._flat_params() if p.requires_grad and self._active(i)}
            if hess_on else {}
        )

        for i in range(q):
            seed_i = self._probe_seeds[i]

            # SPSA scalar c_i along Z_i.
            self._perturb_eps(seed_i, +1.0)
            loss_plus = closure().item()
            if two_side:
                self._perturb_eps(seed_i, -2.0)
                loss_minus = closure().item()
                self._perturb_eps(seed_i, +1.0)  # restore θ
                c_i = (loss_plus - loss_minus) / (2.0 * eps)
            else:
                self._perturb_eps(seed_i, -1.0)  # restore θ
                c_i = (loss_plus - l_best) / eps
            abs_c_sum += abs(c_i)
            samples.append((seed_i, c_i))

            # HiZOO: finite-difference curvature Sᵢ ≈ zᵢᵀ H zᵢ (free — two_side only).
            if hess_on:
                s_i = self._curvature(center, loss_plus, loss_minus, eps)
                curv_abs_sum += abs(s_i)
                for _g, p, idx in self._flat_params():
                    if not p.requires_grad or not self._active(idx):
                        continue
                    z = self._build_z(p, idx, seed_i).float()
                    curv_z2[id(p)].add_(z.mul_(z), alpha=s_i)

            # Evaluate the candidate step L(θ − lr·c_i·Z_i), then restore.
            self._perturb_step(seed_i, c_i, -1.0)
            loss_cand = closure().item()
            self._perturb_step(seed_i, c_i, +1.0)

            if loss_cand < best_probe_loss:
                best_probe_loss = loss_cand
                best_probe_idx = i
            if loss_cand < l_best:
                l_best = loss_cand
                best_seed = seed_i
                best_c = c_i

        self._probe_samples = samples
        self._best_probe_idx = best_probe_idx

        # Directional update with sliding-window acceptance (Lines 16-19).
        window_max = max(self._window) if self._window else float("inf")
        accepted = (best_c is not None) and (l_best <= window_max)
        grad_norm = 0.0
        use_momentum = self.defaults["momentum"] > 0.0
        if use_momentum:
            # Momentum mode: the gate decides what enters the buffer; we always step along it.
            grad_norm = self._momentum_step(best_seed, best_c, accepted)
        elif accepted:
            assert best_seed is not None and best_c is not None
            grad_norm = self._apply_best(best_seed, best_c)

        # Line search: once a step is applied, keep riding the same direction while the
        # (same-batch) loss keeps dropping. Extends along the momentum buffer in momentum
        # mode, otherwise along c_best·Z_best.
        ls_max = self.defaults["line_search_steps"]
        n_extends = 0
        if ls_max > 0 and accepted:
            if use_momentum:
                extend = lambda: self._step_along_momentum(-1.0)
                undo = lambda: self._step_along_momentum(+1.0)
            else:
                extend = lambda: self._perturb_step(best_seed, best_c, -1.0)
                undo = lambda: self._perturb_step(best_seed, best_c, +1.0)
            n_extends = self._line_search(extend, undo, closure, ls_max)

        self._push_window(l_best)

        # HiZOO: one EMA update of the diagonal Hessian from this step's q curvatures
        # (uses the old Σ that pre-conditioned the step; new Σ takes effect next step).
        if hess_on:
            self._hessian_ema(curv_z2, q)
            self._last_metrics["curvature_abs"] = curv_abs_sum / q

        self._step_count += 1
        self._n_steps += 1
        if accepted:
            self._n_accepted += 1

        self._last_metrics["projected_grad_abs"] = abs_c_sum / q
        self._last_metrics["grad_est_norm"] = grad_norm
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        self._last_metrics["line_search_extends"] = float(n_extends)
        self._last_metrics["layer_active_frac"] = self._active_frac

        return center_loss


# ---------------------------------------------------------------------------
# AGZO — Activation-Guided Zeroth-Order  (arXiv:2601.17261)
# ---------------------------------------------------------------------------


class AGZO(ZeroOrderOptimizer):
    """Activation-Guided Zeroth-Order optimization (Lin et al., arXiv:2601.17261).

    Idea: the gradient of a linear layer ``y = x Wᵀ`` is ``∇_W = (∂L/∂y) xᵀ``, so it
    lives in the subspace spanned by the *input activations* ``x``. Instead of isotropic
    Gaussian probes (MeZO), AGZO builds a low-rank activation basis
    ``A_ℓ ∈ R^{d_in×r}`` — the top-r left singular vectors of the layer's activation
    matrix, obtained with a few power-iteration steps on the activation Gram
    ``G_ℓ = Σ_tokens x xᵀ`` captured during the forward pass — and samples the probe as

        ``Δ_ℓ = R_ℓ A_ℓᵀ``,   ``R_ℓ ∈ R^{d_out×r} ~ N(0, I)``      (Eq. 7)

    so the perturbation is confined to where the gradient actually lies → higher expected
    cosine with the true gradient than isotropic probing. Non-linear / non-2-D params, and
    2-D params without a captured basis (e.g. tied embeddings), fall back to plain Gaussian.

    Update — two-point SPSA (Eqs. 8–10), single sample per step:
        ``c = (f(θ+εΔ) − f(θ−εΔ)) / 2ε``   (two_side)
        ``c = (f(θ+εΔ) − f₀) / ε``          (one_side, f₀ = center loss)
        ``θ ← θ − lr · c · Δ``   (+ decoupled weight decay)

    Activation capture: ``register_activation_hooks(model)`` installs forward-pre-hooks on
    the model's ``nn.Linear`` modules. During the *center* forward only (gated by
    ``self._capture``) each hook accumulates ``G_ℓ`` for its weight; power iteration then
    yields ``A_ℓ`` and the full activations are freed immediately — only ``A_ℓ`` (d_in×r,
    r=1 by default) is retained, so the memory overhead is LOZO-level, not first-order.

    DDP: the activation Gram is all-reduced (sum) across ranks and the power iteration is
    seeded identically, so every rank derives the same ``A_ℓ``; ``R_ℓ`` is seeded from the
    broadcast ``z_seed`` and the loss is all-reduced in the closure — hence all ranks apply
    an identical update with no gradient communication (same contract as KronZO). On a
    single GPU no all-reduce happens. FSDP is not supported (blocked in build_optimizer).

    Diagnostic: ``agzo_basis_frac`` (fraction of 2-D params that got an activation basis)
    tells you whether the hooks actually matched the model's linear layers — if it is ~0,
    the ``nn.Linear`` check did not match this architecture and AGZO degenerates to MeZO.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 1,
        power_iters: int = 3,
        step_interval: int = 1,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        layer_sparsity: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        if power_iters < 1:
            raise ValueError(f"Invalid power_iters: {power_iters}")
        if step_interval < 1:
            raise ValueError(f"Invalid step_interval: {step_interval}")
        if perturbation_mode not in ("two_side", "one_side"):
            raise ValueError("perturbation_mode must be 'two_side' or 'one_side'")
        if not (0.0 <= layer_sparsity < 1.0):
            raise ValueError(f"layer_sparsity must be in [0, 1): {layer_sparsity}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            power_iters=power_iters,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            layer_sparsity=layer_sparsity,
        )
        super().__init__(params, defaults)
        self._step_count = 0
        self._last_metrics: dict[str, float] = {}
        # Activation-capture state.
        self._capture = False
        self._act_gram: dict[int, torch.Tensor] = {}   # id(weight) -> accumulated G (d_in×d_in)
        self._act_basis: dict[int, torch.Tensor] = {}  # id(weight) -> basis A (d_in×r)
        self._managed_ids: set[int] = set()            # ids of params this optimizer owns
        self._hook_handles: list = []
        self._last_seed: Optional[int] = None
        # Running |cosine| accumulators for a clean AGZO-vs-Gaussian comparison: a single
        # probe's signed cosine is symmetric around 0 (random sign), so the mean of |cos|
        # is what shows whether the activation-guided probe carries more gradient signal.
        self._cos_abs_sum: dict[str, float] = {"agzo": 0.0, "gauss": 0.0}
        self._cos_n: int = 0
        # LeZO layer sparsity: ids of the param tensors active (perturbed + updated) this
        # step. Empty until _select_active runs; with layer_sparsity=0 it is the full set.
        self._active_wids: set[int] = set()
        self._active_frac: float = 1.0
        for group in self.param_groups:
            for p in group["params"]:
                self._managed_ids.add(id(p))

    def _select_active(self) -> None:
        """LeZO layer sparsity: pick which parameter tensors this step perturbs and updates.

        The active subset rotates deterministically with ``_step_count`` (torch.randperm on
        CPU, identical on every DDP rank — no communication), so over many steps every tensor
        is trained, but each step touches only a ``(1 - layer_sparsity)`` fraction. This cuts
        the per-step perturb/update and activation-capture work and shrinks the probe
        dimension. ``layer_sparsity == 0`` keeps every tensor active (plain AGZO / AGZOKron).
        """
        params = [p for _g, p, _i in self._flat_params() if p.requires_grad]
        n_total = len(params)
        sparsity = self.defaults["layer_sparsity"]
        if sparsity <= 0.0 or n_total == 0:
            self._active_wids = {id(p) for p in params}
            self._active_frac = 1.0
            return
        n_keep = max(1, round(n_total * (1.0 - sparsity)))
        gen = torch.Generator()
        gen.manual_seed((self._step_count * 2_654_435_761 + 12_345) % (2**31))
        perm = torch.randperm(n_total, generator=gen).tolist()
        self._active_wids = {id(params[perm[i]]) for i in range(n_keep)}
        self._active_frac = n_keep / n_total

    def _active(self, p: torch.Tensor) -> bool:
        return id(p) in self._active_wids

    # ------------------------------------------------------------------
    # Activation capture
    # ------------------------------------------------------------------

    def register_activation_hooks(self, model) -> None:
        """Install hooks that capture the input activations of every weight we can score."""
        import torch.nn as nn

        for module in model.modules():
            weight = getattr(module, "weight", None)
            if isinstance(module, nn.Linear) and weight is not None and id(weight) in self._managed_ids:
                self._hook_handles.append(
                    module.register_forward_pre_hook(self._make_hook(id(weight)))
                )

        # Tied LM head. With ``weight_tying`` the output projection is a *functional* call,
        # ``logits = F.linear(ln_f(x), wte.weight)`` (olmo/model.py), so the nn.Linear walk
        # above never sees it and the tied embedding — the single largest tensor in the
        # model, ~52% of all parameters — would fall back to an isotropic Gaussian probe.
        # Its input is exactly the output of ``transformer.ln_f``, so hook that instead.
        # Note this covers the output-projection half of the tied weight's gradient; the
        # embedding-lookup half (sparse rows for the batch's tokens) has no such structure.
        # build_optimizer is called with the *wrapped* model (scripts/train.py passes
        # dist_model), so DDP/FSDP sit between us and the attributes below. ``modules()``
        # sees through the wrapper but attribute access does not: without this unwrap the
        # tied head silently gets no hook, the embedding gets no basis, and its R-space
        # falls back to the full 77M-element shape.
        base = getattr(model, "module", model)
        transformer = getattr(base, "transformer", None)
        config = getattr(base, "config", None)
        if transformer is not None and getattr(config, "weight_tying", False):
            ln_f = getattr(transformer, "ln_f", None)
            wte = getattr(transformer, "wte", None)
            wte_weight = getattr(wte, "weight", None)
            if ln_f is not None and wte_weight is not None and id(wte_weight) in self._managed_ids:
                self._hook_handles.append(
                    ln_f.register_forward_hook(self._make_output_hook(id(wte_weight)))
                )

    def _accumulate_gram(self, wid: int, x) -> None:
        if not self._capture or wid not in self._active_wids:
            return
        if not torch.is_tensor(x) or x.dim() < 1:
            return
        xf = x.reshape(-1, x.shape[-1]).float()  # (N_tokens, d_in)
        g = xf.t() @ xf                          # (d_in, d_in) activation Gram
        prev = self._act_gram.get(wid)
        self._act_gram[wid] = g if prev is None else prev.add_(g)

    def _make_hook(self, wid: int):
        """forward-pre-hook: the module's *input* is the activation we want."""

        def hook(_module, args):
            if args:
                self._accumulate_gram(wid, args[0])

        return hook

    def _make_output_hook(self, wid: int):
        """forward-hook: the module's *output* feeds the matmul we are scoring."""

        def hook(_module, _args, output):
            self._accumulate_gram(wid, output)

        return hook

    def _extract_bases(self) -> None:
        """Power-iterate on each accumulated Gram to get the activation basis A, then free G."""
        try:
            import torch.distributed as dist

            ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            ddp = False

        r = self.defaults["rank"]
        k = self.defaults["power_iters"]
        for wid, G in self._act_gram.items():
            if ddp:
                dist.all_reduce(G, op=dist.ReduceOp.SUM)  # identical basis on every rank
            self._act_basis[wid] = self._power_iter(G, r, k)
        self._act_gram = {}  # free the (large) Gram matrices immediately

    @staticmethod
    def _power_iter(G: torch.Tensor, r: int, k: int) -> torch.Tensor:
        """Top-r eigenvectors of the symmetric PSD matrix G via subspace iteration.

        Deterministic init (fixed seed) so that, given the same (all-reduced) G, every DDP
        rank produces the same orthonormal basis.
        """
        d = G.shape[0]
        r = min(r, d)
        gen = torch.Generator(device=G.device)
        gen.manual_seed(1234567)
        Q = torch.randn(d, r, device=G.device, dtype=torch.float32, generator=gen)
        Q = AGZO._orthonormalize(Q)
        Gf = G.float()
        for _ in range(max(1, k)):
            Q = AGZO._orthonormalize(Gf @ Q)
        return Q  # (d, r), orthonormal, float32

    @staticmethod
    def _orthonormalize(M: torch.Tensor) -> torch.Tensor:
        """Orthonormalize the columns of a tall, very thin matrix (modified Gram-Schmidt).

        Deliberately not ``torch.linalg.qr``: that routes through cuSOLVER, which allocates
        its own handle and workspace, and on a shared GPU with no memory to spare it fails
        outright with ``CUSOLVER_STATUS_INTERNAL_ERROR`` at ``cusolverDnCreate`` — killing
        the run on the very first basis refresh. With ``r`` in the 1..8 range Gram-Schmidt
        is a handful of matmuls, needs no workspace, and is accurate enough for a basis that
        feeds a finite-difference probe. At ``r = 1`` it is just a normalization.
        """
        cols = []
        for j in range(M.shape[1]):
            v = M[:, j]
            for q in cols:  # subtract the components already accounted for
                v = v - q * torch.dot(q, v)
            n = torch.linalg.vector_norm(v)
            cols.append(v / (n + 1e-12))
        return torch.stack(cols, dim=1)

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def _flat_params(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                yield group, p, idx
                idx += 1

    def _get_r_factor(self, d_out: int, r: int, seed: int, idx: int, device, dtype) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed((seed + idx * 1_000_003) % (2**31))
        return torch.randn(d_out, r, device=device, dtype=dtype, generator=gen)

    def _get_gauss(self, shape, seed: int, idx: int, device, dtype) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed((seed + idx * 999_983) % (2**31))
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)

    def _build_delta(self, p: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
        """Δ = R·Aᵀ for a linear weight with a captured activation basis, else Gaussian."""
        if p.dim() == 2:
            A = self._act_basis.get(id(p))
            if A is not None:
                r = A.shape[1]
                R = self._get_r_factor(p.shape[0], r, seed, idx, p.device, p.dtype)
                return R @ A.to(p.device, p.dtype).t()  # (d_out, d_in)
        return self._get_gauss(p.shape, seed, idx, p.device, p.dtype)

    def _perturb(self, seed: int, sign: float) -> None:
        eps = self.defaults["zo_eps"]
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            p.data.add_(self._build_delta(p, idx, seed), alpha=sign * eps)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return AGZO diagnostics from the last step (projected_grad_abs, grad_est_norm,
        agzo_basis_frac — the fraction of 2-D params that received an activation basis)."""
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Cosine of the AGZO probe direction with the true gradient, plus an isotropic
        Gaussian baseline for reference (the paper's core claim: activation-guided > isotropic).

        Must be called right after ``step()`` with ``p.grad`` holding ∇L(θ_k). Returns
        ``cosine_agzo`` (Δ = R·Aᵀ vs ∇L) and ``cosine_gauss`` (isotropic Δ vs ∇L).
        """
        seed = self._last_seed
        if seed is None:
            return {}
        dot_a = norm_a = dot_g = norm_g_probe = norm_grad = 0.0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or p.grad is None or not self._active(p):
                continue
            g = p.grad.float()
            norm_grad += g.pow(2).sum().item()
            da = self._build_delta(p, idx, seed).float()
            dot_a += (g * da).sum().item()
            norm_a += da.pow(2).sum().item()
            dg = self._get_gauss(p.shape, (seed + 7) % (2**31), idx, p.device, torch.float32)
            dot_g += (g * dg).sum().item()
            norm_g_probe += dg.pow(2).sum().item()
        if norm_grad == 0.0:
            return {}
        eps = 1e-12
        c_agzo = dot_a / (math.sqrt(norm_grad * norm_a) + eps)
        c_gauss = dot_g / (math.sqrt(norm_grad * norm_g_probe) + eps)
        # Accumulate |cosine| so the running means give a monotone, sign-free comparison.
        self._cos_n += 1
        self._cos_abs_sum["agzo"] += abs(c_agzo)
        self._cos_abs_sum["gauss"] += abs(c_gauss)
        return {
            "cosine_agzo": c_agzo,
            "cosine_gauss": c_gauss,
            "cosine_agzo_abs": abs(c_agzo),
            "cosine_gauss_abs": abs(c_gauss),
            "cosine_agzo_absmean": self._cos_abs_sum["agzo"] / self._cos_n,
            "cosine_gauss_absmean": self._cos_abs_sum["gauss"] / self._cos_n,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
        self._last_seed = z_seed
        eps = self.defaults["zo_eps"]
        two_side = self.defaults["perturbation_mode"] == "two_side"
        refresh = self._step_count % self.defaults["step_interval"] == 0

        # LeZO: pick this step's active layer subset before capture/perturb/update.
        self._select_active()

        # 1. Center forward at θ_k, capturing activations to (re)build the subspace bases.
        if refresh:
            self._act_gram = {}
            self._capture = True
        center_loss = closure()
        self._capture = False
        if refresh:
            self._extract_bases()
        f0 = center_loss.item()

        # 2. Two-point SPSA scalar along the activation-guided direction Δ.
        self._perturb(z_seed, +1.0)
        f_plus = closure().item()
        if two_side:
            self._perturb(z_seed, -2.0)
            f_minus = closure().item()
            self._perturb(z_seed, +1.0)  # restore θ
            c = (f_plus - f_minus) / (2.0 * eps)
        else:
            self._perturb(z_seed, -1.0)  # restore θ
            c = (f_plus - f0) / eps

        # 3. Apply θ ← θ − lr·c·Δ (+ decoupled weight decay).
        grad_sq = 0.0
        n_2d = n_2d_basis = 0
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            if p.dim() == 2:
                n_2d += 1
                if id(p) in self._act_basis:
                    n_2d_basis += 1
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            delta = self._build_delta(p, idx, z_seed)
            grad_sq += (c * delta.float().norm().item()) ** 2
            p.data.add_(delta, alpha=-lr * c)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)

        self._step_count += 1
        self._last_metrics["projected_grad_abs"] = abs(c)
        self._last_metrics["grad_est_norm"] = math.sqrt(max(grad_sq, 0.0))
        self._last_metrics["agzo_basis_frac"] = (n_2d_basis / n_2d) if n_2d > 0 else 0.0
        self._last_metrics["layer_active_frac"] = self._active_frac
        return center_loss



# ---------------------------------------------------------------------------
# AGZOPlus — AGZO activation basis + SubZero+ multi-query and in-subspace Adam
# ---------------------------------------------------------------------------


class AGZOPlus(_KronDirectional, AGZO):
    """AGZO's activation-guided probe with SubZero+'s multi-query estimation and
    in-subspace Adam.

    The two source methods are orthogonal: AGZO (arXiv:2601.17261) decides *where* to
    sample, SubZero+ (arXiv:2608.15665) decides *how to step* once the subspace is
    fixed.  This class keeps AGZO's choice of subspace and borrows SubZero+'s two
    remaining ingredients.

    **Why the probe stays one-sided.**  The obvious combination — replace SubZero's
    random right factor ``V`` with AGZO's activation basis ``A``, giving
    ``Δ = U Z Aᵀ`` — is a trap.  For a single-query estimator the expected cosine with
    the true gradient factorizes as ``β_D · ‖P_S G‖/‖G‖`` with ``β_D ≈ sqrt(2/(π D))``
    (AGZO Thm. 5.4), and for ``Δ = U Z Aᵀ`` the extra random ``U ∈ R^{d_out×r}`` costs
    exactly what the smaller ``D = r²`` gains::

        β_{r²} · sqrt(r/d_out) · 1  ==  β_{d_out·r}

    i.e. the two-sided variant is worth precisely AGZO at the same rank — and AGZO's own
    rank ablation shows larger rank is *worse*.  So the probe here is AGZO's::

        Δ_ℓ = R_ℓ A_ℓᵀ,   R_ℓ ∈ R^{d_out×r} ~ N(0, I),   A_ℓ ∈ R^{d_in×r} from activations

    What SubZero+ contributes instead:

    1. **Multi-query.**  ``K`` probes ``R⁽ᵏ⁾`` share one basis ``A`` and one baseline
       ``ℓ₀`` (which the activation-capture forward computes anyway, so the ``K+1``
       forward budget costs nothing extra)::

           ρ_k = (ℓ_k − ℓ₀)/ε,      R̄ = (1/K) Σ_k ρ_k R⁽ᵏ⁾ ∈ R^{d_out×r}

       Averaging inside an *aligned* subspace is worth far more than inside a random
       one: as ``K → ∞`` this converges to ``P_A(G)``, and ``‖P_A G‖/‖G‖ ≈ 1`` because
       the gradient's row space lies in the activation span — whereas SubZero's random
       subspace has a hard ceiling at ``r/sqrt(m·n)``.

    2. **Adam in coefficient space.**  Moments live in ``R^{d_out×r}``, the space ``R̄``
       lives in — ``2·d_out·r`` floats per layer against ``2·d_out·d_in`` for full-space
       Adam, a factor ``r/d_in`` (≈1/900 at r=1)::

           M ← β₁M + (1−β₁)R̄,   V ← β₂V + (1−β₂)(R̄ ⊙ R̄)
           W ← W − lr · (M̂ / (sqrt(V̂) + eps)) Aᵀ

    **Rank stays 1.**  With ``K`` queries the reachable cosine is
    ``≈ min(sqrt(K)·β_{d_out·r}·energy(r), energy(r))``.  Raising ``r`` only pays once
    ``K`` approaches the saturation point ``K ≈ π·d_out·r/2`` (≈1400 at d_out=896,
    r=1) — far out of reach — while ``energy(r)`` is already ~0.9 at r=1.  So r=1
    remains optimal here, exactly as in single-query AGZO.

    **Moment remapping is not optional here.**  ``A`` is rebuilt from the current
    minibatch every ``step_interval`` steps (default: every step), so the coefficient
    coordinates change under the Adam buffers — and ``_power_iter`` gives no sign
    guarantee, so even at r=1 the basis can flip sign between steps and silently negate
    the momentum.  Hence ``moment_policy`` defaults to ``project`` (change of basis
    ``M ← M (A_oldᵀ A_new)``), not to SubZero+'s paper-faithful ``keep``.

    **KronZO selection.**  ``kron_query_budget = q > 0`` makes q trials per step, each a
    full K-probe estimate (+ Adam from the current moments) on the shared basis, and applies
    only the best candidate step (see ``_KronDirectional``).  With ``K = 1`` and
    ``update_rule="sgd"`` this is ``AGZOKron`` with the one_side baseline fixed to the centre.

    1-D params, and 2-D params with no captured basis, fall back to a plain Gaussian
    probe with element-wise full-space Adam (negligible: they are a tiny fraction of d).

    DDP: identical contract to ``AGZO`` — the activation Gram is all-reduced, the power
    iteration is seeded identically, ``R⁽ᵏ⁾`` comes from the broadcast ``z_seed``, and the
    loss is all-reduced in the closure, so all ranks apply the same update.  FSDP is not
    supported (blocked in ``build_optimizer``).
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 1,
        power_iters: int = 3,
        step_interval: int = 1,
        num_queries: int = 8,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        perturbation_mode: str = "one_side",
        weight_decay: float = 0.0,
        layer_sparsity: float = 0.0,
        update_rule: str = "adam",
        moment_policy: str = "project",
        kron_query_budget: int = 0,
        kron_history_length: int = 10,
    ):
        if num_queries < 1:
            raise ValueError(f"Invalid num_queries: {num_queries}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if update_rule not in ("adam", "sgd"):
            raise ValueError("update_rule must be 'adam' or 'sgd'")
        if moment_policy not in ("keep", "zero", "project"):
            raise ValueError("moment_policy must be 'keep', 'zero' or 'project'")

        super().__init__(
            params,
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            power_iters=power_iters,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            layer_sparsity=layer_sparsity,
        )
        for group in self.param_groups:
            group.setdefault("num_queries", num_queries)
            group.setdefault("betas", betas)
            group.setdefault("eps", eps)
            group.setdefault("update_rule", update_rule)
            group.setdefault("moment_policy", moment_policy)
        self.defaults.update(
            num_queries=num_queries,
            betas=betas,
            eps=eps,
            update_rule=update_rule,
            moment_policy=moment_policy,
        )
        self._init_kron(kron_query_budget, kron_history_length)
        self._adam_steps = 0  # committed Adam updates; drives bias correction

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return AGZOPlus diagnostics from the last step.

        Metrics
        -------
        projected_grad_abs, projected_grad_std
            Mean ``|ρ_k|`` and the spread of ``ρ_k`` over the ``K`` queries; their ratio
            is an SNR proxy for whether ``K`` is large enough (see the metric pages).
        subspace_grad_norm
            Global norm of the aggregated coefficient matrix ``R̄``.  ``A`` is
            column-orthonormal, so this equals ``‖R̄ Aᵀ‖_F`` — the full-space gradient
            estimate norm, at ``d_out×r`` cost.
        grad_est_norm
            Norm of the actual update direction before ``lr``.
        agzo_basis_frac
            Fraction of 2-D params that received an activation basis.  If this is ~0 the
            hooks did not match this architecture and the method degenerates to
            multi-query MeZO with Adam.
        layer_active_frac
            LeZO active-layer fraction (1.0 unless ``layer_sparsity`` > 0).
        update_accepted, accept_rate
            KronZO selection diagnostics; only reported with ``kron_query_budget > 0``.
            In that mode ``subspace_grad_norm`` / ``grad_est_norm`` describe the selected
            trial (0 on a rejected step) and ``projected_grad_*`` cover all ``q·K`` probes.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    # ------------------------------------------------------------------
    # Probe pieces
    # ------------------------------------------------------------------

    def _coeff(self, p: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
        """The probe's *coefficient* tensor in float32: ``R`` (d_out×r) if this param has
        an activation basis, else the dense Gaussian ``z`` of the param's own shape."""
        if p.dim() == 2 and id(p) in self._act_basis:
            r = self._act_basis[id(p)].shape[1]
            return self._get_r_factor(p.shape[0], r, seed, idx, p.device, torch.float32)
        return self._get_gauss(p.shape, seed, idx, p.device, torch.float32)

    def _query_seed(self, z_seed: int, k: int) -> int:
        return (z_seed + k * 987_654_321) % (2**31)

    def _remap_moments(self, old_bases: dict[int, torch.Tensor]) -> None:
        """Re-express the Adam buffers after the activation bases were rebuilt.

        The coefficient coordinates are tied to ``A``: column ``j`` of ``R`` weights the
        direction ``A[:, j]``.  Once ``A`` changes those coordinates mean something else,
        and because ``_power_iter`` fixes no sign convention, at r=1 the basis vector can
        simply flip, turning accumulated momentum into anti-momentum.

        * ``keep``    — do nothing (SubZero+'s paper-faithful behaviour; here it is the
                        risky option, since the basis moves every step by default).
        * ``zero``    — restart Adam in the new basis.
        * ``project`` — change of basis ``M ← M (A_oldᵀ A_new)``, exact for the component
                        that survives into the new subspace (and at r=1 this is exactly
                        the sign/overlap correction).  The second moment is not a linear
                        object, so rotating it is meaningless; its mean is broadcast,
                        which preserves the per-layer step scale and drops stale
                        per-coordinate structure.
        """
        policy = self.defaults["moment_policy"]
        if policy == "keep":
            return
        for _group, p, _idx in self._flat_params():
            state = self.state[p]
            if "exp_avg" not in state:
                continue
            wid = id(p)
            a_old, a_new = old_bases.get(wid), self._act_basis.get(wid)
            if a_old is None or a_new is None:
                continue  # dense probe: coordinates are full-space, nothing moved
            if policy == "zero" or a_old.shape != a_new.shape:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                continue
            state["exp_avg"] = state["exp_avg"] @ (a_old.T @ a_new)
            state["exp_avg_sq"] = torch.full_like(
                state["exp_avg_sq"], state["exp_avg_sq"].mean().item()
            )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
        self._last_seed = z_seed
        eps = self.defaults["zo_eps"]
        num_queries = self.defaults["num_queries"]
        two_side = self.defaults["perturbation_mode"] == "two_side"
        refresh = self._step_count % self.defaults["step_interval"] == 0

        self._select_active()

        # 1. Centre forward at θ: gives the shared baseline ℓ₀ AND captures activations.
        if refresh:
            self._act_gram = {}
            self._capture = True
        center_loss = closure()
        self._capture = False
        if refresh:
            old_bases = dict(self._act_basis)
            self._extract_bases()
            self._remap_moments(old_bases)

        # agzo_basis_frac: did the hooks actually produce a basis for the active weights?
        n_2d = n_2d_basis = 0
        for _group, p, _idx in self._flat_params():
            if p.requires_grad and self._active(p) and p.dim() == 2:
                n_2d += 1
                n_2d_basis += int(id(p) in self._act_basis)
        self._last_metrics["agzo_basis_frac"] = (n_2d_basis / n_2d) if n_2d > 0 else 0.0

        if self._kron_q > 0:
            scalars = self._kron_select_step(closure, center_loss, z_seed, num_queries, two_side, eps)
        else:
            # 2. K probes sharing the basis; one_side reuses ℓ₀ so the cost is K+1 forwards.
            scalars, _ = self._estimate(closure, center_loss, z_seed, num_queries, two_side, eps)
            self._apply_update(scalars, z_seed)

        rho = list(scalars.values())
        mean_rho = sum(rho) / len(rho)
        self._step_count += 1
        self._last_metrics["projected_grad_abs"] = sum(abs(s) for s in rho) / len(rho)
        self._last_metrics["projected_grad_std"] = (
            math.sqrt(sum((s - mean_rho) ** 2 for s in rho) / len(rho)) if len(rho) > 1 else 0.0
        )
        self._last_metrics["layer_active_frac"] = self._active_frac
        return center_loss

    def _estimate(
        self,
        closure,
        center_loss: torch.Tensor,
        z_seed: int,
        num_queries: int,
        two_side: bool,
        eps: float,
    ) -> tuple[dict[int, float], None]:
        """K probes on the shared basis → SPSA scalars."""
        f0 = center_loss.item()
        scalars: dict[int, float] = {}
        for k in range(num_queries):
            seed_k = self._query_seed(z_seed, k)
            self._perturb(seed_k, +1.0)
            f_plus = closure().item()
            if two_side:
                self._perturb(seed_k, -2.0)
                f_minus = closure().item()
                self._perturb(seed_k, +1.0)  # restore θ
                scalars[k] = (f_plus - f_minus) / (2.0 * eps)
            else:
                self._perturb(seed_k, -1.0)  # restore θ
                scalars[k] = (f_plus - f0) / eps
        return scalars, None

    def _apply_update(self, scalars: dict[int, float], z_seed: int) -> None:
        dirs = self._compute_directions(scalars, z_seed, commit=True)
        self._move(dirs, -1.0)
        self._decay(dirs)

    def _compute_directions(self, scalars: dict[int, float], z_seed: int, commit: bool) -> list:
        """Aggregate ``R̄`` and return the coefficient-space step directions
        ``[(group, p, coeff), ...]`` (before ``lr``); nothing is applied.  With
        ``commit=True`` the Adam state advances; with ``commit=False`` it is left untouched
        and the direction is what a committed call would produce from the same state."""
        num_queries = len(scalars)
        t = self._adam_steps + 1  # Adam bias-correction step, 1-based
        grad_sq = 0.0
        subspace_sq = 0.0
        dirs = []

        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            beta1, beta2 = group["betas"]
            adam_eps = group["eps"]

            # R̄ = (1/K) Σ_k ρ_k R⁽ᵏ⁾ — coefficients only, never the full d_out×d_in probe.
            r_bar = torch.zeros_like(self._coeff(p, idx, self._query_seed(z_seed, 0)))
            for k, rho in scalars.items():
                r_bar.add_(self._coeff(p, idx, self._query_seed(z_seed, k)), alpha=rho)
            r_bar.div_(num_queries)
            # A column-orthonormal ⇒ ‖R̄ Aᵀ‖_F = ‖R̄‖_F.
            subspace_sq += r_bar.norm().item() ** 2

            if group["update_rule"] == "adam":
                state = self.state[p]
                exp_avg, exp_avg_sq = state.get("exp_avg"), state.get("exp_avg_sq")
                if exp_avg is None or exp_avg.shape != r_bar.shape:
                    exp_avg, exp_avg_sq = torch.zeros_like(r_bar), torch.zeros_like(r_bar)
                    if commit:
                        state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
                if commit:
                    exp_avg.mul_(beta1).add_(r_bar, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(r_bar, r_bar, value=1.0 - beta2)
                else:
                    exp_avg = exp_avg.mul(beta1).add(r_bar, alpha=1.0 - beta1)
                    exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(r_bar, r_bar, value=1.0 - beta2)
                m_hat = exp_avg / (1.0 - beta1**t)
                v_hat = exp_avg_sq / (1.0 - beta2**t)
                direction = m_hat / (v_hat.sqrt() + adam_eps)
            else:
                direction = r_bar

            grad_sq += direction.norm().item() ** 2
            dirs.append((group, p, direction))

        if commit:
            self._adam_steps += 1
        self._last_metrics["subspace_grad_norm"] = math.sqrt(max(subspace_sq, 0.0))
        self._last_metrics["grad_est_norm"] = math.sqrt(max(grad_sq, 0.0))
        return dirs

    # ------------------------------------------------------------------
    # Moves (shared with the KronZO selection)
    # ------------------------------------------------------------------

    def _lift(self, p: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        """Coefficients → full space: ``c Aᵀ`` for a weight with a basis, identity otherwise."""
        if p.dim() == 2 and id(p) in self._act_basis:
            A = self._act_basis[id(p)].to(device=p.device, dtype=torch.float32)
            return coeff @ A.T
        return coeff

    def _move(self, dirs: list, sign: float) -> None:
        """θ += sign · lr · lift(coeff)."""
        for group, p, coeff in dirs:
            p.data.add_(self._lift(p, coeff).to(p.dtype), alpha=sign * group["lr"])

    def _decay(self, dirs: list) -> None:
        for group, p, _coeff in dirs:
            if group["weight_decay"] != 0.0:
                p.data.mul_(1.0 - group["lr"] * group["weight_decay"])


# ---------------------------------------------------------------------------
# AGZOKron — AGZO perturbation + KronZO directional selection
# ---------------------------------------------------------------------------


class AGZOKron(AGZO):
    """AGZO activation-guided perturbation + KronZO directional 'lucky-RGE' update.

    Inherits AGZO's activation capture and ``Δ = R·Aᵀ`` probe (shared per-layer activation
    basis ``A``, fresh random ``R`` per probe) and replaces AGZO's single-sample SPSA update
    with KronZO's directional selection:

      1. Sample ``q`` activation-guided probes ``Δ_i`` (same ``A``, different ``R_i``).
      2. For each, form the SPSA scalar ``c_i`` and evaluate the candidate step
         ``L(θ − lr·c_i·Δ_i)``.
      3. Keep the single best-decreasing direction (not the average).
      4. Apply it only if its loss beats the worst of the last ``h`` accepted losses
         (sliding-window acceptance); otherwise skip the step.

    Why combine them: a single AGZO probe has larger |cosine| with the true gradient than an
    isotropic one but random sign (the ``R`` factor). The lucky-RGE selection cherry-picks the
    probe that actually lowers the loss — correct sign *and* large magnitude — turning AGZO's
    amplitude advantage into an aligned descent step. Cost: 1 + 3q forward passes (two_side),
    the +1 being the capture forward. DDP / activation capture behave exactly as in AGZO.

    ``layer_sparsity`` (LeZO-style layer sparsity, see AGZO) composes orthogonally on top:
    setting it turns this class into what the project calls "AGZO-Kron-LeZO" — no separate
    class is needed, it is this same optimizer with ``agzo_layer_sparsity > 0``.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 1,
        power_iters: int = 3,
        step_interval: int = 1,
        query_budget: int = 8,
        history_length: int = 10,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        layer_sparsity: float = 0.0,
        update_direction: str = "best",
    ):
        if query_budget < 1:
            raise ValueError(f"Invalid query_budget: {query_budget}")
        if history_length < 0:
            raise ValueError(f"Invalid history_length: {history_length}")
        if update_direction not in ("best", "qrge"):
            raise ValueError(f"update_direction must be 'best' or 'qrge', got {update_direction!r}")
        super().__init__(
            params,
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            power_iters=power_iters,
            step_interval=step_interval,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            layer_sparsity=layer_sparsity,
        )
        self.defaults["query_budget"] = query_budget
        self.defaults["history_length"] = history_length
        self.defaults["update_direction"] = update_direction
        # Sliding acceptance window (prefilled with +inf so the first h steps always pass).
        self._window: list[float] = [float("inf")] * history_length
        self._n_steps = 0
        self._n_accepted = 0
        # Per-step probe bookkeeping for the ZO-vs-FO cosine diagnostic.
        self._probe_seeds: list[int] = []
        self._probe_samples: list[tuple[int, float]] = []
        self._best_probe_idx: int = -1
        # (seed, candidate-step loss) per probe — a fitness-style diagnostic an ES/RL
        # layer on top of this optimizer can read (AGZOKronRL collects its own returns).
        self._probe_cand_losses: list[tuple[int, float]] = []

    # ------------------------------------------------------------------
    # Directional-update helpers (activation-guided Δ, per-group lr)
    # ------------------------------------------------------------------

    def _qrge_direction(self, p, idx: int, samples: list[tuple[int, float]]):
        """``(1/q)·Σ c_i Δ_i`` for one parameter — the plain multi-query RGE estimate.

        Built in fp32 one parameter at a time, so the extra memory is a single tensor
        rather than a second copy of the model.
        """
        acc = None
        for seed_i, c_i in samples:
            d = self._build_delta(p, idx, seed_i).float().mul_(c_i)
            acc = d if acc is None else acc.add_(d)
        return acc.div_(len(samples))

    def _perturb_qrge(self, samples: list[tuple[int, float]], sign: float) -> None:
        """θ += sign · lr · (1/q)·Σ c_i Δ_i   (trial step along the averaged direction)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            p.data.add_(self._qrge_direction(p, idx, samples).to(p.dtype),
                        alpha=sign * group["lr"])

    def _apply_qrge(self, samples: list[tuple[int, float]]) -> float:
        """Apply θ -= lr · (1/q)·Σ c_i Δ_i (+ decoupled weight decay); return update norm."""
        grad_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            d = self._qrge_direction(p, idx, samples)
            grad_sq += d.norm().item() ** 2
            p.data.add_(d.to(p.dtype), alpha=-lr)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
        return math.sqrt(max(grad_sq, 0.0))

    def _perturb_step(self, seed: int, c: float, sign: float) -> None:
        """θ += sign · lr · c · Δ   (candidate / trial step)."""
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            p.data.add_(self._build_delta(p, idx, seed), alpha=sign * group["lr"] * c)

    def _apply_best(self, seed: int, c: float) -> float:
        """Apply θ -= lr·c·Δ (+ decoupled weight decay); return the update Frobenius norm."""
        grad_sq = 0.0
        for group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p):
                continue
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            delta = self._build_delta(p, idx, seed)
            grad_sq += (c * delta.float().norm().item()) ** 2
            p.data.add_(delta, alpha=-lr * c)
            if weight_decay != 0.0:
                p.data.mul_(1.0 - lr * weight_decay)
        return math.sqrt(max(grad_sq, 0.0))

    def _push_window(self, loss: float) -> None:
        if not self._window:
            return
        j = max(range(len(self._window)), key=lambda k: self._window[k])
        self._window[j] = loss

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fo_cosine_metrics(self) -> dict[str, float]:
        """Cosine of the selected ('lucky') AGZO direction and of the q-RGE average vs ∇L."""
        samples = self._probe_samples
        if not samples:
            return {}
        q = len(samples)
        # In q-RGE mode no probe is selected, so cosine_selected is undefined — but
        # cosine_qrge is then exactly the direction stepped along, which is the whole
        # point of the comparison, so the metric must not be suppressed with it.
        has_selected = self._best_probe_idx >= 0
        best_seed, best_c = samples[self._best_probe_idx] if has_selected else (None, None)
        dot_sel = norm_sel = dot_avg = norm_avg = norm_g = 0.0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or p.grad is None or not self._active(p):
                continue
            g = p.grad.float()
            norm_g += g.pow(2).sum().item()
            if has_selected:
                sel = self._build_delta(p, idx, best_seed).float().mul_(best_c)
                dot_sel += (g * sel).sum().item()
                norm_sel += sel.pow(2).sum().item()
            avg = torch.zeros_like(g)
            for seed_i, c_i in samples:
                avg.add_(self._build_delta(p, idx, seed_i).float(), alpha=c_i)
            avg.div_(q)
            dot_avg += (g * avg).sum().item()
            norm_avg += avg.pow(2).sum().item()
        if norm_g == 0.0:
            return {}
        e = 1e-12
        out = {"cosine_qrge": dot_avg / (math.sqrt(norm_g * norm_avg) + e)}
        if has_selected:
            out["cosine_selected"] = dot_sel / (math.sqrt(norm_g * norm_sel) + e)
        return out

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
        self._last_seed = z_seed
        eps = self.defaults["zo_eps"]
        q = self.defaults["query_budget"]
        two_side = self.defaults["perturbation_mode"] == "two_side"
        qrge = self.defaults["update_direction"] == "qrge"
        refresh = self._step_count % self.defaults["step_interval"] == 0

        # Fix this step's q probe seeds up front (each → a fresh R factor, shared basis A).
        self._probe_seeds = [(z_seed + i * 987_654_321) % (2**31) for i in range(q)]

        # LeZO: pick this step's active layer subset before capture/probe/update.
        self._select_active()

        # 1. Center forward + activation capture → per-layer bases A.
        if refresh:
            self._act_gram = {}
            self._capture = True
        center_loss = closure()
        self._capture = False
        if refresh:
            self._extract_bases()
        l_best = center_loss.item()
        best_seed: Optional[int] = None
        best_c: Optional[float] = None
        abs_c_sum = 0.0
        samples: list[tuple[int, float]] = []
        cand_losses: list[tuple[int, float]] = []
        best_probe_idx = -1
        best_probe_loss = float("inf")

        # 2. q activation-guided probes; keep the best-decreasing candidate step.
        for i in range(q):
            seed_i = self._probe_seeds[i]
            self._perturb(seed_i, +1.0)
            loss_plus = closure().item()
            if two_side:
                self._perturb(seed_i, -2.0)
                loss_minus = closure().item()
                self._perturb(seed_i, +1.0)  # restore θ
                c_i = (loss_plus - loss_minus) / (2.0 * eps)
            else:
                self._perturb(seed_i, -1.0)  # restore θ
                c_i = (loss_plus - l_best) / eps
            abs_c_sum += abs(c_i)
            samples.append((seed_i, c_i))

            # The per-probe candidate evaluation exists only to rank probes against each
            # other. q-RGE averages all of them, so ranking is pointless — skipping it
            # drops one forward per probe (1+3q → 1+2q+1 with the averaged trial below).
            if qrge:
                continue

            # Evaluate the candidate step L(θ − lr·c_i·Δ_i), then restore.
            self._perturb_step(seed_i, c_i, -1.0)
            loss_cand = closure().item()
            self._perturb_step(seed_i, c_i, +1.0)
            cand_losses.append((seed_i, loss_cand))

            if loss_cand < best_probe_loss:
                best_probe_loss = loss_cand
                best_probe_idx = i
            if loss_cand < l_best:
                l_best = loss_cand
                best_seed = seed_i
                best_c = c_i

        self._probe_samples = samples
        self._probe_cand_losses = cand_losses
        self._best_probe_idx = best_probe_idx

        # 3. Sliding-window acceptance, then apply the best direction.
        window_max = max(self._window) if self._window else float("inf")
        grad_norm = 0.0
        if qrge:
            # One trial along the averaged direction, so the sliding window plays exactly
            # the role it does for best-of-q and the *only* thing that changed is which
            # direction we step along.
            self._perturb_qrge(samples, -1.0)
            l_best = closure().item()
            self._perturb_qrge(samples, +1.0)
            accepted = l_best <= window_max
            if accepted:
                grad_norm = self._apply_qrge(samples)
        else:
            accepted = (best_c is not None) and (l_best <= window_max)
            if accepted:
                assert best_seed is not None and best_c is not None
                grad_norm = self._apply_best(best_seed, best_c)
        self._push_window(l_best)

        self._step_count += 1
        self._n_steps += 1
        if accepted:
            self._n_accepted += 1

        n_2d = sum(1 for _g, p, _i in self._flat_params() if p.requires_grad and p.dim() == 2)
        n_basis = sum(
            1
            for _g, p, _i in self._flat_params()
            if p.requires_grad and p.dim() == 2 and id(p) in self._act_basis
        )
        self._last_metrics["projected_grad_abs"] = abs_c_sum / q
        self._last_metrics["grad_est_norm"] = grad_norm
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        self._last_metrics["agzo_basis_frac"] = (n_basis / n_2d) if n_2d > 0 else 0.0
        self._last_metrics["layer_active_frac"] = self._active_frac
        return center_loss


# ---------------------------------------------------------------------------
# AGZOKronRL — AGZO-Kron with a learned mean on the R factor (ZO-RL, "P1")
# ---------------------------------------------------------------------------

class AGZOKronRL(AGZOKron):
    """AGZO-Kron with a learned per-layer mean on the activation-guided probe factor.

    "P1" from the AGZO note. AGZO factorizes the probe as ``Δ = R·Aᵀ`` with an analytic
    input-side basis ``A`` (top-r activation directions, see :class:`AGZO`) and a random
    output/mixing factor ``R ~ N(0, I)``. AGZOKronRL keeps ``A`` analytic but learns the
    *mean* of ``R`` online with a ZO-RL score-function (evolution-strategies) update, so
    the sampler is biased toward historically low-loss directions while the activation
    subspace keeps the input side aligned.

    Per 2-D weight with a captured basis we store ``M ∈ R^{d_out×r}`` (init ``zero`` /
    ``random`` / from a first-order gradient projected onto ``A``). Each probe draws
    ``R_i ~ N(M, σ²)``, ``Δ_i = R_i·Aᵀ``; KronZO's best-of-q selection + sliding-window
    acceptance (inherited from :class:`AGZOKron`) pick and apply the step. After the step
    the ES advantage update shifts ``M`` toward the low-loss probes:

        ``a_i = (q·f_i − Σ_j f_j)/(q−1)``,
        ``g_M = −(1/(q·σ²)) Σ_i a_i·(M − R_i)``,   ``M ← M − η_μ·g_M``,

    where ``f_i`` is the return of probe ``i`` (``perturbed`` = ``L(θ+εΔ_i)`` or
    ``candidate`` = ``L(θ−η c_i Δ_i)``). ``M`` is frozen within the step (all ``R_i``
    regenerated from the step's probe seeds with the same ``M``) and updated once at the
    end, so DDP ranks stay in sync (same ``z_seed``, all-reduced losses, identical ``M``).
    Memory stays AGZO-level: ``M`` is ``d_out×r`` per layer (``r=1`` → one d_out-vector
    per layer), no full-parameter buffer.

    ``σ`` (``variance``) matches the ZO-RL convention: it is the standard deviation of the
    probe factor, so ``variance=1`` at ``M=0`` reproduces plain AGZO's ``R ~ N(0, I)``.

    Mean-update rule (``update``):

    * ``es`` — score-function / evolution-strategies gradient above. The per-probe weight
      (``fitness``) is either the mean-subtracted ``advantage`` ``a_i = (q f_i − Σ f_j)/(q−1)``
      or centered ``rank`` utilities (lowest loss → −0.5 … highest → +0.5), which are
      invariant to loss scale and robust to outliers (NES / OpenAI-ES fitness shaping).
    * ``cem`` — Cross-Entropy Method: pick the ``elite_frac`` fraction of probes with the
      lowest loss and move ``M`` toward the elite mean ``R``: ``M ← M + η_μ(R̄_elite − M)``
      (``η_μ = 1`` is a full CEM refit; smaller values smooth it). No ``σ²`` scaling.

    ``mirrored`` draws the ``R`` noise in antithetic pairs (``M ± σζ``) — a near-free
    variance reducer for the mean update. It relies on the return being **sign-aware**:
    with ``mu_return='candidate'`` (symmetric in ``Δ ↦ −Δ``) the paired contributions
    cancel, so ``perturbed`` is the sensible choice (a warning is emitted, not enforced).

    FO init (``agzo_rl_mu_init_from_fo``): the Trainer runs one first-order forward
    (capturing activations → bases) + backward (grads), then :meth:`init_mu_from_fo_grad`
    seeds ``M = ∇_W·A`` — the FO gradient projected onto the activation basis — so
    ``Δ = M·Aᵀ = P_A ∇_W`` ≈ ∇_W on the first step (``P_A = A Aᵀ``). See ``olmo/train.py``.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        rank: int = 1,
        power_iters: int = 3,
        step_interval: int = 1,
        query_budget: int = 8,
        history_length: int = 10,
        variance: float = 1.0,
        lr_mu: Optional[float] = None,
        mu_return: str = "perturbed",
        mu_init: str = "zero",
        update: str = "es",
        fitness: str = "advantage",
        mirrored: bool = False,
        elite_frac: float = 0.5,
        perturbation_mode: str = "two_side",
        weight_decay: float = 0.0,
        layer_sparsity: float = 0.0,
    ):
        if variance <= 0:
            raise ValueError(f"variance (σ) must be > 0, got {variance}")
        if mu_return not in ("perturbed", "candidate"):
            raise ValueError(f"mu_return must be 'perturbed' or 'candidate', got {mu_return!r}")
        if mu_init not in ("zero", "random"):
            raise ValueError(f"mu_init must be 'zero' or 'random', got {mu_init!r}")
        if update not in ("es", "cem"):
            raise ValueError(f"update must be 'es' or 'cem', got {update!r}")
        if fitness not in ("advantage", "rank"):
            raise ValueError(f"fitness must be 'advantage' or 'rank', got {fitness!r}")
        if not (0.0 < elite_frac <= 1.0):
            raise ValueError(f"elite_frac must be in (0, 1], got {elite_frac}")
        if mirrored and mu_return == "candidate":
            warnings.warn(
                "AGZOKronRL: mirrored=True with mu_return='candidate' — the candidate loss is "
                "symmetric in Δ↦−Δ, so antithetic pairs cancel and the mean update ≈ 0. "
                "Use mu_return='perturbed' for a sign-aware signal.",
                stacklevel=2,
            )
        super().__init__(
            params,
            lr=lr,
            zo_eps=zo_eps,
            rank=rank,
            power_iters=power_iters,
            step_interval=step_interval,
            query_budget=query_budget,
            history_length=history_length,
            perturbation_mode=perturbation_mode,
            weight_decay=weight_decay,
            layer_sparsity=layer_sparsity,
        )
        self._sigma = float(variance)  # std σ of R ~ N(M, σ²) (ZO-RL convention)
        self._lr_mu = lr_mu if (lr_mu is not None and lr_mu >= 0.0) else lr
        self._mu_return = mu_return
        self._mu_init = mu_init
        self._update = update
        self._fitness = fitness
        self._mirrored = mirrored
        self._elite_frac = elite_frac
        # Per-step mirrored-sampling maps (probe seed -> shared noise seed / sign);
        # populated in step(). Empty defaults make _randn_r behave as plain i.i.d.
        self._noise_seed: dict[int, int] = {}
        self._mirror_sign: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Learned mean M on the R factor
    # ------------------------------------------------------------------

    def _randn_r(self, seed: int, idx: int, d_out: int, r: int, device, dtype) -> torch.Tensor:
        """Standard-normal R noise for one probe, reproducible from (seed, idx).

        With mirrored sampling the noise seed and sign come from the per-step maps built in
        ``step()`` so antithetic partners share ``ζ`` and differ only by sign; without them
        (empty maps) it falls back to plain i.i.d. noise keyed by the probe seed.
        """
        base = self._noise_seed.get(seed, seed)
        sign = self._mirror_sign.get(seed, 1.0)
        gen = torch.Generator(device=device)
        gen.manual_seed((base + idx * 1_000_003) % (2**31))
        z = torch.randn(d_out, r, device=device, dtype=dtype, generator=gen)
        return z if sign >= 0.0 else z.neg_()

    def _mean(self, p: torch.Tensor, r: int) -> torch.Tensor:
        """Per-weight learnable mean ``M ∈ R^{d_out×r}`` (float32), lazily initialised."""
        state = self.state[p]
        M = state.get("M")
        if M is None or tuple(M.shape) != (p.shape[0], r):
            if self._mu_init == "random":
                M = torch.randn(p.shape[0], r, dtype=torch.float32, device=p.device)
                n = M.norm()
                M = (M / n) if n > 0 else M
            else:  # "zero"
                M = torch.zeros(p.shape[0], r, dtype=torch.float32, device=p.device)
            state["M"] = M
        return state["M"]

    def _build_delta(self, p: torch.Tensor, idx: int, seed: int) -> torch.Tensor:
        """Δ = R·Aᵀ with R ~ N(M, σ²) for a linear weight with a basis; else Gaussian."""
        if p.dim() == 2:
            A = self._act_basis.get(id(p))
            if A is not None:
                r = A.shape[1]
                M = self._mean(p, r).to(p.dtype)
                R = M + self._sigma * self._randn_r(seed, idx, p.shape[0], r, p.device, p.dtype)
                return R @ A.to(p.device, p.dtype).t()
        return self._get_gauss(p.shape, seed, idx, p.device, p.dtype)

    @staticmethod
    def _fitness_weights(f: torch.Tensor, kind: str) -> torch.Tensor:
        """Per-probe weights for the ES update (lower loss → more negative weight).

        ``advantage``: mean-subtracted returns ``(q f_i − Σ f_j)/(q−1)``.
        ``rank``: centered ranks — sort by ascending loss, map to ``[−0.5, +0.5]`` so the
        best probe gets −0.5 and the worst +0.5 (scale-free, robust to outliers).
        """
        q = f.numel()
        if kind == "rank":
            if q < 2:
                return torch.zeros_like(f)
            order = torch.argsort(f)  # ascending: order[0] = lowest loss = best
            ranks = torch.empty_like(f)
            ranks[order] = torch.arange(q, dtype=f.dtype)
            return ranks / (q - 1) - 0.5
        return (f * q - f.sum()) / max(q - 1, 1)

    def _update_mean(self, returns: list) -> float:
        """One update of every active layer's mean ``M`` (dispatch on ``update``).

        Regenerates each probe's ``R_i`` from the step's frozen ``M`` and probe seeds (so
        it matches the R used during sampling, including mirrored signs). For ``es`` it is
        the score-function step ``M ← M − η_μ·g_M`` with ``g_M = −(1/(qσ²)) Σ a_i (M − R_i)``
        and weights ``a_i`` from :meth:`_fitness_weights`. For ``cem`` it moves ``M`` toward
        the mean ``R`` of the lowest-loss ``elite_frac`` fraction of probes.
        """
        q = len(returns)
        f = torch.tensor(returns, dtype=torch.float32)
        sigma2 = self._sigma * self._sigma
        seeds = [s for s, _c in self._probe_samples]

        if self._update == "cem":
            k = max(1, round(q * self._elite_frac))
            elite = torch.topk(f, k, largest=False).indices.tolist()  # k lowest losses
        else:
            coeff = self._fitness_weights(f, self._fitness)

        mu_grad_sq = 0.0
        mu_norm_sum = 0.0
        n_layers = 0
        for _group, p, idx in self._flat_params():
            if not p.requires_grad or not self._active(p) or p.dim() != 2:
                continue
            A = self._act_basis.get(id(p))
            if A is None:
                continue
            r = A.shape[1]
            M = self._mean(p, r)  # float32 (d_out, r)

            if self._update == "cem":
                # Move M toward the elite mean R: ΔM = η_μ (R̄_elite − M), R̄ = M + σ·mean(ζ_elite).
                z_sum = torch.zeros_like(M)
                for i in elite:
                    z_sum.add_(self._randn_r(seeds[i], idx, p.shape[0], r, p.device, torch.float32))
                step = z_sum.mul_(self._sigma / len(elite))  # R̄_elite − M
                M.add_(step, alpha=self._lr_mu)
                mu_grad_sq += (step.norm().item() * self._lr_mu) ** 2
            else:
                m_diff = torch.zeros_like(M)
                for i, seed_i in enumerate(seeds):
                    R = M + self._sigma * self._randn_r(seed_i, idx, p.shape[0], r, p.device, torch.float32)
                    m_diff.add_(M - R, alpha=coeff[i].item())
                g_M = m_diff.neg_().div_(q * sigma2)  # −(1/(qσ²)) Σ a_i (M − R_i)
                M.add_(g_M, alpha=-self._lr_mu)
                mu_grad_sq += g_M.norm().item() ** 2

            mu_norm_sum += M.norm().item()
            n_layers += 1
        n = max(n_layers, 1)
        self._last_metrics["avg_mu_norm"] = mu_norm_sum / n
        self._last_metrics["avg_mu_grad_norm"] = math.sqrt(max(mu_grad_sq, 0.0)) / n
        return math.sqrt(max(mu_grad_sq, 0.0))

    # ------------------------------------------------------------------
    # FO-gradient initialisation of M (driven by the Trainer)
    # ------------------------------------------------------------------

    def begin_fo_mu_capture(self) -> None:
        """Arm activation capture over ALL managed weights for a one-off FO init.

        Called by the Trainer right before the single first-order forward-backward used
        to seed ``M``. Marks every managed weight active so the forward hooks fire for
        all linear layers (``_select_active`` on the first real step resets this).
        """
        self._act_gram = {}
        self._active_wids = set(self._managed_ids)
        self._active_frac = 1.0
        self._capture = True

    @torch.no_grad()
    def init_mu_from_fo_grad(self, normalize: bool = False) -> dict:
        """Seed ``M`` from a first-order gradient projected onto the activation basis.

        Call after :meth:`begin_fo_mu_capture` + one FO forward-backward: the forward
        fills the activation Grams (→ bases), the backward fills ``.grad``. Sets
        ``M = (scale·∇_W)·A`` per 2-D weight with a basis, so ``Δ = M·Aᵀ = P_A ∇_W`` ≈
        the true gradient on the first step. ``normalize`` rescales by the global FO
        gradient norm before projecting. Returns ``{fo_grad_norm, normalized}``.
        """
        self._capture = False
        # Detach the captured Grams (the FO forward had autograd on) and build bases.
        self._act_gram = {k: v.detach() for k, v in self._act_gram.items()}
        self._extract_bases()

        grads: dict[int, torch.Tensor] = {}
        sq = 0.0
        for _group, p, _idx in self._flat_params():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                grads[id(p)] = g
                sq += g.pow(2).sum().item()
        fo_norm = math.sqrt(sq)
        scale = (1.0 / fo_norm) if (normalize and fo_norm > 0.0) else 1.0

        for _group, p, _idx in self._flat_params():
            if not p.requires_grad or p.dim() != 2:
                continue
            A = self._act_basis.get(id(p))
            g = grads.get(id(p))
            if A is None or g is None:
                continue
            r = A.shape[1]
            M = self._mean(p, r)  # (d_out, r) float32
            M.copy_((g * scale) @ A.float())  # project FO grad onto activation basis A
        return {"fo_grad_norm": fo_norm, "normalized": float(normalize)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))
        self._last_seed = z_seed
        eps = self.defaults["zo_eps"]
        q = self.defaults["query_budget"]
        two_side = self.defaults["perturbation_mode"] == "two_side"
        refresh = self._step_count % self.defaults["step_interval"] == 0

        self._probe_seeds = [(z_seed + i * 987_654_321) % (2**31) for i in range(q)]
        # Mirrored sampling: pair consecutive probes (0,1),(2,3),… to share the noise seed
        # with opposite sign (R = M ± σζ); a trailing odd probe stays unpaired. Empty maps
        # (mirrored off) make _randn_r fall back to plain i.i.d. noise per probe seed.
        if self._mirrored:
            self._noise_seed = {}
            self._mirror_sign = {}
            for i, s in enumerate(self._probe_seeds):
                self._noise_seed[s] = self._probe_seeds[i - (i % 2)]  # even partner's seed
                self._mirror_sign[s] = 1.0 if (i % 2 == 0) else -1.0
        else:
            self._noise_seed = {}
            self._mirror_sign = {}
        self._select_active()

        # 1. Center forward + activation capture → per-layer bases A.
        if refresh:
            self._act_gram = {}
            self._capture = True
        center_loss = closure()
        self._capture = False
        if refresh:
            self._extract_bases()
        l_best = center_loss.item()
        best_seed: Optional[int] = None
        best_c: Optional[float] = None
        abs_c_sum = 0.0
        samples: list[tuple[int, float]] = []
        returns: list[float] = []  # f_i feeding the ES M-update
        best_probe_idx = -1
        best_probe_loss = float("inf")

        # 2. q learned-mean activation-guided probes; keep the best-decreasing candidate.
        for i in range(q):
            seed_i = self._probe_seeds[i]
            self._perturb(seed_i, +1.0)
            loss_plus = closure().item()
            if two_side:
                self._perturb(seed_i, -2.0)
                loss_minus = closure().item()
                self._perturb(seed_i, +1.0)  # restore θ
                c_i = (loss_plus - loss_minus) / (2.0 * eps)
            else:
                self._perturb(seed_i, -1.0)  # restore θ
                c_i = (loss_plus - l_best) / eps
            abs_c_sum += abs(c_i)
            samples.append((seed_i, c_i))

            self._perturb_step(seed_i, c_i, -1.0)
            loss_cand = closure().item()
            self._perturb_step(seed_i, c_i, +1.0)

            returns.append(loss_cand if self._mu_return == "candidate" else loss_plus)
            if loss_cand < best_probe_loss:
                best_probe_loss = loss_cand
                best_probe_idx = i
            if loss_cand < l_best:
                l_best = loss_cand
                best_seed = seed_i
                best_c = c_i

        self._probe_samples = samples
        self._best_probe_idx = best_probe_idx

        # 3. Sliding-window acceptance, then apply the best direction.
        window_max = max(self._window) if self._window else float("inf")
        accepted = (best_c is not None) and (l_best <= window_max)
        grad_norm = 0.0
        if accepted:
            assert best_seed is not None and best_c is not None
            grad_norm = self._apply_best(best_seed, best_c)
        self._push_window(l_best)

        # 4. ES score-function update of the learned mean M (uses the frozen per-step M).
        self._update_mean(returns)

        self._step_count += 1
        self._n_steps += 1
        if accepted:
            self._n_accepted += 1

        n_2d = sum(1 for _g, p, _i in self._flat_params() if p.requires_grad and p.dim() == 2)
        n_basis = sum(
            1
            for _g, p, _i in self._flat_params()
            if p.requires_grad and p.dim() == 2 and id(p) in self._act_basis
        )
        self._last_metrics["projected_grad_abs"] = abs_c_sum / q
        self._last_metrics["grad_est_norm"] = grad_norm
        self._last_metrics["update_accepted"] = 1.0 if accepted else 0.0
        self._last_metrics["accept_rate"] = self._n_accepted / self._n_steps
        self._last_metrics["agzo_basis_frac"] = (n_basis / n_2d) if n_2d > 0 else 0.0
        self._last_metrics["layer_active_frac"] = self._active_frac
        return center_loss


# ---------------------------------------------------------------------------
# HiZOO
# ---------------------------------------------------------------------------

class HiZOO(ZeroOrderOptimizer):
    """Hessian-Informed Zeroth-Order Optimizer (HiZOO / HiZOO).

    Implements "Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed
    Zeroth-Order Optimizer" (Zhao, Dang, Ye, Dai, Qian, Tsang, ICLR 2025),
    Algorithm 1 (see ``papers/HiZO.pdf``).

    Idea
    ----
    MeZO uses an isotropic Gaussian probe ``u`` and takes a step of fixed scale
    regardless of the local curvature.  Heterogeneous curvature across LLM
    parameters makes this converge slowly / towards saddle points.  HiZOO
    additionally estimates a **diagonal Hessian** with one extra forward pass and
    uses it as a pre-conditioner, so each coordinate is scaled by its curvature —
    a diagonal-Newton step obtained from forward passes only.

    Per parameter we store the diagonal ``Σ⁻¹`` (an estimate of the diagonal
    Hessian, initialised to ``1``), EMA-updated each step.  ``Σ^{1/2} = 1/√Σ⁻¹``
    pre-conditions both the perturbation and the descent direction.

    Each step (Algorithm 1) costs **3 forward passes** and, per parameter,
    ``u ~ N(0, I)`` reproduced from the shared seed:

      1. ``ℓ   = L(θ)``                                   (center)
      2. ``ℓ₊  = L(θ + μ Σ^{1/2} u)``
      3. ``ℓ₋  = L(θ − μ Σ^{1/2} u)``
      4. scalar curvature ``S = (ℓ₊ + ℓ₋ − 2ℓ) / (2μ²)`` and
         projected gradient ``g = (ℓ₊ − ℓ₋) / (2μ)``.
      5. diagonal Hessian estimate ``diag(Σ'ₜ) = S · Σ⁻¹ₜ₋₁ · u²`` and EMA update
         ``Σ⁻¹ₜ = (1−α) Σ⁻¹ₜ₋₁ + α |diag(Σ'ₜ)|`` (``α`` = smooth scale, Eq. 5).
      6. descent (Eq. 1/2), using the **updated** ``Σₜ``:
         ``θᵢ ← θᵢ − ηₜ · g · Σ^{1/2}ₜ,ᵢ · uᵢ``.

    Memory: one extra ``O(d)`` float32 buffer (the diagonal Hessian).  The probe
    ``u`` is regenerated from the seed rather than stored, matching MeZO's
    memory-frugal pattern.

    DDP: ``u`` is seeded from the externally broadcast ``z_seed`` and the loss is
    all-reduced inside the closure, so every rank derives identical scalars, an
    identical Hessian state and an identical update without gradient
    communication.  FSDP is not supported (blocked in ``build_optimizer``).

    Notes
    -----
    * The paper's Algorithm 1 is inherently three-pass (the Hessian needs the
      center loss ``ℓ``), so ``perturbation_mode`` is ignored here.
    * ``num_pert_samples = 1`` (``n = 1``) is the paper's default; larger ``n``
      averages several probes per step (HiZOO-multi) for a lower-variance Hessian
      estimate at the cost of ``1 + 2n`` forward passes.
    """

    def __init__(
        self,
        params,
        lr: float,
        zo_eps: float = 1e-3,
        smooth_scale: float = 1e-6,
        hessian_min: float = 1e-8,
        weight_decay: float = 0.0,
        vector_sampling_type: str = "standard_normal",
        num_pert_samples: int = 1,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if zo_eps <= 0:
            raise ValueError(f"Invalid zo_eps: {zo_eps}")
        if not (0.0 <= smooth_scale <= 1.0):
            raise ValueError(f"smooth_scale (EMA α) must be in [0, 1], got {smooth_scale}")
        if hessian_min <= 0:
            raise ValueError(f"hessian_min must be > 0, got {hessian_min}")
        if num_pert_samples < 1:
            raise ValueError(f"Invalid num_pert_samples: {num_pert_samples}")

        defaults = dict(
            lr=lr,
            zo_eps=zo_eps,
            smooth_scale=smooth_scale,
            hessian_min=hessian_min,
            weight_decay=weight_decay,
            num_pert_samples=num_pert_samples,
        )
        super().__init__(params, defaults)
        self.vector_sampler = VectorSampler(vector_sampling_type)
        self._generators: dict[torch.device, torch.Generator] = {}
        self._last_metrics: dict[str, float] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]

    def _reset_generators(self, z_seed: int) -> None:
        """Seed each device's generator once before iterating parameters.

        See MeZO._reset_generators for the rationale (seeding once lets the
        generator advance sequentially so each parameter gets an independent
        probe, while the same seed reproduces the same probes later).
        """
        seen: set[torch.device] = set()
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.device not in seen:
                    self._get_generator(p.device).manual_seed(z_seed)
                    seen.add(p.device)

    def _hess(self, p: torch.Tensor) -> torch.Tensor:
        """Return the per-parameter diagonal Hessian estimate ``Σ⁻¹`` (float32).

        Lazily initialised to ones (``Σ₀ = I``) on first access.
        """
        state = self.state[p]
        if "hess" not in state:
            state["hess"] = torch.ones_like(p, dtype=torch.float32)
        return state["hess"]

    def get_post_step_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Return HiZOO diagnostics collected during the last step.

        Metrics
        -------
        projected_grad_abs
            Absolute SPSA scalar ``|(ℓ₊ − ℓ₋) / (2μ)|`` (mean over ``n`` probes).
        curvature_abs
            Absolute finite-difference curvature ``|(ℓ₊ + ℓ₋ − 2ℓ) / (2μ²)|``
            (mean over ``n`` probes) — the raw signal driving the Hessian EMA.
        grad_est_norm
            Global L2 norm of the applied pre-conditioned update direction
            ``g · Σ^{1/2} · u`` across all parameters (before ``lr`` scaling).
        hessian_mean
            Mean of the diagonal Hessian estimate ``Σ⁻¹`` across all parameters
            (1.0 at init; drifts towards the local curvature scale).
        precond_sqrt_mean
            Mean pre-conditioner ``Σ^{1/2} = 1/√Σ⁻¹`` — the effective per-coordinate
            step-scale multiplier relative to MeZO.
        """
        return {k: torch.tensor(v) for k, v in self._last_metrics.items()}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], z_seed: Optional[int] = None) -> torch.Tensor:
        if z_seed is None:
            z_seed = int(np.random.randint(0, 1_000_000_000))

        mu = self.defaults["zo_eps"]
        num_pert = self.defaults["num_pert_samples"]

        # Center loss ℓ = L(θ). One evaluation shared by all n probes.
        center_loss = closure()
        center = center_loss.item()

        samples: list[tuple[int, float, float]] = []
        pg_abs_sum = 0.0
        curv_abs_sum = 0.0
        for i in range(num_pert):
            seed_i = (z_seed + i * 987_654_321) % (2**31)

            self._perturb(seed_i, scaling_factor=+1.0)   # θ + μ Σ^{1/2} u
            loss_plus = closure().item()
            self._perturb(seed_i, scaling_factor=-2.0)   # θ − μ Σ^{1/2} u
            loss_minus = closure().item()
            self._perturb(seed_i, scaling_factor=+1.0)   # restore θ (unmoved during sampling)

            projected_grad = (loss_plus - loss_minus) / (2.0 * mu)
            curvature = (loss_plus + loss_minus - 2.0 * center) / (2.0 * mu * mu)
            samples.append((seed_i, projected_grad, curvature))
            pg_abs_sum += abs(projected_grad)
            curv_abs_sum += abs(curvature)

        # All n probes were evaluated at the SAME θ (restored after each pair), and
        # against the SAME fresh center ℓ. Form a single averaged Hessian-EMA update
        # and a single averaged pre-conditioned descent step (HiZOO-multi, Eq. 4/1),
        # so larger n *reduces variance* instead of taking n sequential micro-steps.
        self._apply_update(samples)

        self._last_metrics["projected_grad_abs"] = pg_abs_sum / num_pert
        self._last_metrics["curvature_abs"] = curv_abs_sum / num_pert
        return center_loss

    def _perturb(self, z_seed: int, scaling_factor: float) -> None:
        """θ ← θ + scaling_factor · μ · Σ^{1/2} · u, with ``Σ^{1/2} = 1/√Σ⁻¹``.

        Uses the *current* (pre-update) Hessian, so the ±μ / −2μ sequence cancels
        exactly and restores θ.
        """
        self._reset_generators(z_seed)
        mu = self.defaults["zo_eps"]
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                gen = self._get_generator(p.device)
                u = self.vector_sampler.sample(p.shape, p.device, gen).float()
                precond = torch.rsqrt(self._hess(p))  # Σ^{1/2} = 1/√Σ⁻¹
                delta = u.mul_(scaling_factor * mu).mul_(precond)
                p.data.add_(delta.to(p.dtype))

    def _apply_update(self, samples: list[tuple[int, float, float]]) -> None:
        """One averaged Hessian-EMA update + one averaged pre-conditioned descent step.

        Every probe in ``samples`` was evaluated at the same θ. Per parameter we
        average over the ``n`` probes (Eq. 4 / Eq. 1), reproducing each ``uᵢ`` from
        its seed:

            curv_u2 = (1/n) Σᵢ curvᵢ · uᵢ²      (drives the diagonal Hessian)
            grad_u  = (1/n) Σᵢ gᵢ   · uᵢ        (the descent direction)

        then update the Hessian once (old Σ⁻¹ pre-conditions the estimate, Eq. 4/5):

            diag(Σ'ₜ) = Σ⁻¹ₜ₋₁ · curv_u2 ,  Σ⁻¹ₜ = (1−α) Σ⁻¹ₜ₋₁ + α |diag(Σ'ₜ)|

        and take one step with the **updated** Σₜ:  ``θ ← θ − lr · grad_u · Σ^{1/2}ₜ``.

        For ``n = 1`` this is identical to Algorithm 1. For ``n > 1`` the averaging
        is the variance-reducing HiZOO-multi variant (Appendix D.2): the estimator's
        variance falls ~1/n while θ moves once, not n times.
        """
        n = len(samples)
        alpha = self.defaults["smooth_scale"]
        hess_min = self.defaults["hessian_min"]

        # Per-parameter accumulators over the n probes (Σᵢ curvᵢ·uᵢ² and Σᵢ gᵢ·uᵢ).
        curv_u2: dict[int, torch.Tensor] = {}
        grad_u: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    curv_u2[id(p)] = torch.zeros_like(p, dtype=torch.float32)
                    grad_u[id(p)] = torch.zeros_like(p, dtype=torch.float32)

        for seed_i, projected_grad, curvature in samples:
            self._reset_generators(seed_i)
            for group in self.param_groups:
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    gen = self._get_generator(p.device)
                    u = self.vector_sampler.sample(p.shape, p.device, gen).float()
                    curv_u2[id(p)].add_(u.mul(u), alpha=curvature)
                    grad_u[id(p)].add_(u, alpha=projected_grad)

        grad_sum_sq = 0.0
        hess_sum = 0.0
        precond_sum = 0.0
        n_elems = 0
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                hess = self._hess(p)  # Σ⁻¹ₜ₋₁ (float32, in-place buffer)

                # Averaged diagonal Hessian estimate and one EMA update (Eq. 4/5).
                sigma_prime = hess.mul(curv_u2[id(p)]).div_(n)  # Σ⁻¹ · (1/n)Σ curvᵢ uᵢ²
                hess.mul_(1.0 - alpha).add_(sigma_prime.abs_(), alpha=alpha)
                hess.clamp_(min=hess_min)

                # One averaged pre-conditioned descent step with the updated Σₜ.
                precond = torch.rsqrt(hess)  # Σ^{1/2}ₜ = 1/√Σ⁻¹ₜ
                grad_est = grad_u[id(p)].div_(n).mul_(precond)
                grad_sum_sq += grad_est.pow(2).sum().item()
                if weight_decay != 0.0:
                    grad_est.add_(p.data.float(), alpha=weight_decay)
                p.data.add_(grad_est.to(p.dtype), alpha=-lr)

                hess_sum += hess.sum().item()
                precond_sum += precond.sum().item()
                n_elems += hess.numel()

        self._last_metrics["grad_est_norm"] = math.sqrt(max(grad_sum_sq, 0.0))
        if n_elems > 0:
            self._last_metrics["hessian_mean"] = hess_sum / n_elems
            self._last_metrics["precond_sqrt_mean"] = precond_sum / n_elems
