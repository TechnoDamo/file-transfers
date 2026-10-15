from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError
from .util import StrEnum

__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "InitFnType",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "InstanceFilterConfig",
    "EvaluatorConfig",
    "TokenizerConfig",
    "TrainConfig",
    "PaddingDirection",
    "TruncationDirection",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "DDPConfig",
    "DistributedStrategy",
    "DDPGradSyncMode",
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "SingleGPUConfig",
    "CheckpointType",
    "DATA_ROOT_ENV_VAR",
    "expand_data_path",
    "expand_data_paths",
]

DATA_ROOT_ENV_VAR = "OLMO_DATA_ROOT"
"""
Environment variable used as ``TrainConfig.data_root`` when the config does not set it,
so that ``${data_root}`` in data paths works without editing the config per machine.
"""

DATA_SHARD_SUFFIX = ".npy"
"""
Suffix of the memmap shards that are picked up when a data path points at a directory.
"""


def _natural_sort_key(path: str) -> Tuple[Union[int, str], ...]:
    """
    Sort key that orders digit runs numerically, so that e.g. ``chunk-9`` comes before
    ``chunk-10`` even when the shard numbers are not zero-padded to the same width.
    """
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path))


def expand_data_path(path: str, validate: bool = True) -> List[str]:
    """
    Expand a single data path from a config into the list of shards it stands for:
    a local directory becomes the ``*.npy`` shards under it, sorted; a single shard and
    a URL are passed through untouched.

    ``${data_root}`` in the path has already been substituted by OmegaConf by the time
    this runs -- see ``TrainConfig.data_root``.
    """
    from .util import is_url

    if is_url(path):
        return [path]

    dir_path = Path(path)
    if dir_path.is_dir():
        shards = sorted((str(shard) for shard in dir_path.glob(f"**/*{DATA_SHARD_SUFFIX}")), key=_natural_sort_key)
        if not shards and validate:
            raise OLMoConfigurationError(f"No '*{DATA_SHARD_SUFFIX}' shards found in directory '{path}'")
        return shards

    return [path]


def expand_data_paths(paths: Optional[List[str]], validate: bool = True) -> Optional[List[str]]:
    """
    :func:`expand_data_path` over a list of paths, flattened.
    """
    if paths is None:
        return None
    out: List[str] = []
    for path in paths:
        out.extend(expand_data_path(path, validate=validate))
    return out


C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            from .util import is_url

            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        # Finds the latest checkpoint in a folder.
        def path_last_checkpoint(path) -> str:
            from .util import find_latest_checkpoint

            latest_checkpoint = find_latest_checkpoint(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(f"Could not find a latest checkpoint at {path}")
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver("path.last_checkpoint", path_last_checkpoint, replace=True)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        return config

    @classmethod
    def _pre_resolve(cls, config: DictConfig) -> None:
        """
        Hook called on the merged config *before* OmegaConf resolves its interpolations,
        so it can still fill in values that other nodes interpolate. No-op by default.
        """
        del config

    def _post_load(self, validate_paths: bool = True) -> None:
        """
        Hook called on the instantiated config right after it has been built by
        :meth:`load` or :meth:`new`, with interpolations already resolved. No-op by default.
        """
        del validate_paths

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            cls._pre_resolve(cast(DictConfig, conf))
            config = cast(C, om.to_object(conf))
            config._post_load()
            return config
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))
            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            cls._pre_resolve(cast(DictConfig, conf))
            config = cast(C, om.to_object(conf))
            config._post_load(validate_paths=validate_paths)
            return config
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out

    def update_with(self, **kwargs):
        result = deepcopy(self)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


@dataclass
class ModelConfig(BaseConfig):
    """
    OLMo (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for Llama2-style grouped query attention.
    """

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.
    """

    alibi: bool = False
    """
    If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.
    """

    alibi_bias_max: float = 8.0
    """
    Maximum absolute value of ALiBi bias.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: int = 10_000
    """
    The theta setting for RoPE.
    """

    flash_attention: bool = False
    """
    If ``True``, use ``FlashAttention``.
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    multi_query_attention: Optional[bool] = None
    """
    Deprecated. Use n_kv_heads instead.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    embedding_layer_norm: bool = False
    """
    Apply layer norm directly to the embeddings.
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    layer_norm_eps: float = 1e-05

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    In PaLM, they got rid of all bias terms because they found that large
    models tend to have near 0 bias terms anyway.
    """

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in
    layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    scale_logits: bool = False
    """
    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    weight_tying: bool = True
    """
    Whether to tie output linear weights to the input embedding.
    """

    eos_token_id: int = 50256
    """
    The ID of the end-of-sentence special token.
    """

    pad_token_id: int = 50256
    """
    The ID of the token to use for padding. Defaults to the ID of the EOS token.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_fn: InitFnType = InitFnType.normal
    """
    The weight initialization strategy.
    """

    init_std: float = 0.02
    """
    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """

    scale_emb_init: bool = False
    """
    If ``True``, embeddings are scaled up by ``sqrt(d_model)`` during initialization.
    Currently this is only used with `full_megatron` init when ``emb_init_std`` is unset.
    """

    emb_init_std: Optional[float] = None
    """
    Override the standard deviation to use when initializing the embedding weights.
    """

    norm_after: bool = False
    """
    Apply norm after the attention/feedforward layers rather than before, as introduced in the Swin transformer paper (Liu et al).
    """

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise OLMoConfigurationError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"
    mezo = "mezo"
    lezo = "lezo"
    zo_adam = "zo_adam"
    zo_sign_sgd = "zo_sign_sgd"
    lozo = "lozo"
    subzero_plus = "subzero_plus"
    zo_muon = "zo_muon"
    kron_zo = "kron_zo"
    hizoo = "hizoo"
    agzo = "agzo"
    agzo_plus = "agzo_plus"
    agzo_kron = "agzo_kron"
    agzo_kron_rl = "agzo_kron_rl"
    ldsd_muon = "ldsd_muon"
    ldsd_sign_sgd = "ldsd_sign_sgd"
    ldsd_rl = "ldsd_rl"
    ldsd_rl_adamm = "ldsd_rl_adamm"
    ldsd_rl_sgd = "ldsd_rl_sgd"
    ldsd_rl_kron = "ldsd_rl_kron"
    oracle_probe = "oracle_probe"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.lionw
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5

    no_decay_norm_and_bias: Optional[bool] = None
    """
    Deprecated. Use ``decay_norm_and_bias`` and ``decay_embeddings`` instead.
    """

    selective_updates: bool = False
    """
    If ``True``, optimizer parameter and state updates are skipped when the corresponding gradient is 0.
    """

    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    metrics_log_interval: Optional[int] = None
    """
    The interval with which to collect and log detailed parameter-specific metrics.
    This only applies when logging to W&B, since these metrics won't be logged to the console.
    If not set, defaults to the wandb `log_interval`.
    """

    record_update_metrics: bool = False
    """
    Whether to record detailed metrics about the optimizer's parameter updates, like the norm and max
    of the update with AdamW.
    """

    # --- Zero-order (MeZO / ZO family) ---
    zo_eps: float = 1e-3
    """Perturbation scale ε for zero-order SPSA estimates."""

    zo_perturbation_mode: str = "two_side"
    """``two_side`` (±ε) or ``one_side`` (+ε then restore)."""

    zo_fo_cosine_interval: int = 0
    """Steps between first-order probe-quality diagnostics (0 = off).

    On a firing step the trainer runs one extra forward+backward at θ_k to populate
    ``.grad``, then calls the optimizer's ``fo_cosine_metrics()`` and logs the cosine
    between the probe direction and the true gradient. That costs a full first-order
    step, so keep the interval well above 1; 100 reproduces the sampling density of the
    historical AGZO runs.

    Optimizers without ``fo_cosine_metrics`` skip the extra backward entirely.
    """

    oracle_probe_sign: str = "exact"
    """Where ``oracle_probe`` gets the descent sign.

    ``exact`` (default) computes ``sign(∇L·z)`` directly and runs no probe forwards, so
    ``zo_eps`` drops out of the instrument entirely. ``spsa`` recovers the sign from a
    two-point finite difference like a real ZO method — faithful, but at production
    ``zo_eps`` the difference is a chord rather than a derivative and degrades as the
    probe becomes better aligned, which inverts the cosine ladder.
    """

    oracle_probe_update: str = "matched"
    """How ``oracle_probe`` turns its two-point measurement into a step.

    ``matched`` (default) keeps the measured sign but uses the update magnitude an
    isotropic probe would have produced, so every rung of the cosine ladder takes the
    same size of step and only the direction varies. ``spsa`` uses the raw estimate,
    whose magnitude grows like ``c·√d`` — that confounds the ladder with a step-size
    sweep and diverges within ten steps at c=0.03.
    """

    oracle_probe_cosine: float = 0.0
    """Target cosine between probe and true gradient for the ``oracle_probe`` optimizer.

    ``oracle_probe`` is an instrument rather than a method — it needs a real backward
    every step. Sweeping this value with everything else fixed measures how much probe
    alignment a learned sampler would have to deliver to move the loss at all. ``0.0``
    reproduces isotropic MeZO and is the control arm.
    """

    mezo_momentum: float = 0.0
    """Momentum for MeZO / ZO-SGD updates (0 = plain ZO-SGD)."""

    mezo_vector_sampling_type: str = "standard_normal"
    """Probe distribution: ``standard_normal`` or ``lp_sphere``."""

    mezo_num_pert_samples: int = 1
    """Number of independent SPSA probe pairs averaged per step."""

    lezo_layer_sparsity: float = 0.0
    """LeZO (arXiv:2410.09823): the paper's sparse rate ρ — the fraction of parameter
    tensors ("layers") frozen each step (must be in [0, 1); 0 degenerates to plain
    ``mezo``). Semantics are identical to ``agzo_layer_sparsity`` / ``kronzo_layer_sparsity``
    — same rotation rule, same per-tensor granularity — so the three are directly
    comparable. ``LeZO`` reuses every other MeZO knob (``zo_eps``, ``zo_perturbation_mode``,
    ``mezo_momentum``, ``mezo_vector_sampling_type``, ``mezo_num_pert_samples``) and adds
    only this one: each step perturbs and updates just the active
    ``(1 - lezo_layer_sparsity)`` subset, which rotates deterministically with the step count
    (identical on every DDP rank, no communication), so all parameters are still trained over
    time. Two wins: less perturb/update work per step (the paper reports >3x speedup at
    ρ=0.75), and a probe in a lower dimension d_active, which lifts the per-step ZO/FO
    alignment (~1/√d_active). Typical: 0.5-0.75. Note: the paper finds higher sparsity needs
    a *higher* learning rate — re-tune ``learning_rate`` when you change this."""

    lozo_rank: int = 4
    """Low-rank factor rank for LOZO 2D weight probes."""

    lozo_step_interval: int = 1
    """Refresh LOZO V factors every this many optimizer steps."""

    lozo_num_pert_samples: int = 1
    """Number of LOZO SPSA probe pairs averaged per step."""

    # --- KronZO directional selection for SubZeroPlus / AGZOPlus ---

    zo_kron_query_budget: int = 0
    """SubZeroPlus / AGZOPlus: KronZO directional selection with q = this many trials per step
    (0 = off). Each trial is a full host estimate — ``subzero_num_queries`` /
    ``agzo_plus_num_queries`` (K) probes averaged, turned into a step by the host's update rule
    (Adam from the current moments, or SGD) — and its candidate step is evaluated; only the best
    one is applied (subject to the acceptance window) and only its moments are committed.
    Cost 1+q(K+1) forwards (one_side) / 1+q(2K+1) (two_side). K=1 with ``*_update_rule: sgd``
    is the ``agzo_kron`` scheme."""

    zo_kron_history_length: int = 10
    """SubZeroPlus / AGZOPlus with ``zo_kron_query_budget > 0``: acceptance window h — the best candidate
    must also beat the worst of the last h accepted losses. 0 disables it. Same meaning and
    default as ``agzo_history_length`` / ``kronzo_history_length``."""

    # --- AGZOPlus (AGZO basis + SubZero+ multi-query / subspace Adam) ---

    agzo_plus_num_queries: int = 8
    """AGZOPlus: number of probes K averaged per step, all sharing one activation basis and
    one baseline. Costs K+1 forwards (the baseline forward is the activation-capture pass,
    which AGZO runs anyway). Averaging inside an *aligned* subspace converges towards the
    true gradient, unlike SubZero+'s random subspace which has a hard ceiling."""

    agzo_plus_update_rule: str = "adam"
    """AGZOPlus: ``adam`` runs Adam in the d_out×r coefficient space; ``sgd`` steps along
    ``R̄ Aᵀ`` directly (= plain multi-query AGZO)."""

    agzo_plus_moment_policy: str = "project"
    """AGZOPlus: what to do with the Adam buffers when the activation basis is rebuilt.
    ``project`` (default) applies the change of basis ``M ← M (A_oldᵀ A_new)``; ``zero``
    restarts Adam; ``keep`` carries stale coordinates over. Unlike SubZero+, the basis here
    moves every ``agzo_step_interval`` steps (default: every step) AND ``_power_iter`` fixes
    no sign convention, so at r=1 a sign flip would negate the momentum — hence ``keep`` is
    NOT a safe default here. Keep ``agzo_rank`` at 1: raising it only pays once K approaches
    ~π·d_out·r/2, which is out of reach, while the captured gradient energy is already ~0.9
    at r=1 (see the AGZO rank ablation)."""

    # --- SubZero+ (arXiv:2608.15665) ---

    subzero_rank: int = 24
    """SubZero+: subspace rank r. Perturbations for a 2-D weight live in the r²-dimensional
    space spanned by ``U Z Vᵀ``, so the estimator variance is O(r²) instead of O(m·n) and the
    Adam buffers are r×r per layer. Paper uses r = 24..64."""

    subzero_step_interval: int = 50
    """SubZero+: redraw the projections U, V every F steps. Smaller F explores more directions
    but invalidates the Adam buffers more often (see ``subzero_moment_policy``)."""

    subzero_num_queries: int = 8
    """SubZero+: number of perturbations K averaged per step, all sharing the same subspace.
    Costs K+1 forwards per step (or 2K with ``zo_perturbation_mode: two_side``). The paper uses
    K = 99 and reports that K ≤ 10 is too noisy to train — but it also spends the whole budget on
    400 optimizer steps, so treat the K/steps split as the main thing to tune here."""

    subzero_update_rule: str = "adam"
    """SubZero+: ``adam`` runs Adam inside the r×r subspace; ``sgd`` steps along ``U Z̄ Vᵀ``
    directly (= SubZero with multi-query). The paper's own tables have SGD competitive with or
    ahead of Adam on average, so both are worth sweeping."""

    subzero_haar: bool = True
    """SubZero+: apply the ``sgn(R_ii)`` correction to the QR factor so U, V are Haar-distributed
    and directionally unbiased. Costs O(r) sign flips; set False only for ablations."""

    subzero_moment_policy: str = "keep"
    """SubZero+: what to do with the r×r Adam buffers when U, V are redrawn. ``keep`` is the
    paper's behaviour (buffers carry over even though the coordinates now mean different
    directions); ``zero`` restarts Adam in the new basis; ``project`` re-expresses the first
    moment via ``M ← (Uₙᵀ Uₒ) M (Vₒᵀ Vₙ)`` and flattens the second moment to its mean. Only
    matters when the run is long relative to ``subzero_step_interval``."""

    zo_muon_rank: int = 4
    """Subspace rank for ZOMuon Haar projection P."""

    zo_muon_step_interval: int = 1
    """Refresh ZOMuon projection P every this many steps."""

    zo_muon_num_samples: int = 1
    """Number of ZOMuon subspace samples (1 → two-sided antithetic)."""

    zo_muon_ns_steps: int = 5
    """Newton–Schulz iterations for ZOMuon orthogonalization."""

    # --- KronZO (Kronecker-factored zero-order, GERAD G-2025-44) ---

    kronzo_step_interval: int = 50
    """KronZO: how often the frozen Kronecker factor B is refreshed (ν). 1 = every step."""

    kronzo_query_budget: int = 10
    """KronZO: number of probe directions q evaluated per step. Cost is
    1 + 3q forward passes (two_side) or 1 + 2q (one_side). The paper uses q = 10..50."""

    kronzo_history_length: int = 10
    """KronZO: length h of the sliding loss window for the directional-update acceptance
    test. The step is applied only if the best probe loss beats the worst loss in the
    window. 0 disables the test (always accept)."""

    kronzo_orthogonal_probes: bool = False
    """KronZO: QR-orthogonalize the q probe directions within each step so they are
    mutually orthogonal (maximizes coverage of the B-induced subspace per step) instead
    of i.i.d. Gaussian. Cheap (orthogonalizes the small A factors). Combine with
    kronzo_step_interval=1 to also vary the subspace every step."""

    kronzo_momentum: float = 0.0
    """KronZO: heavy-ball momentum β on the ZO gradient estimate (0 disables). The chosen
    per-step direction is integrated into a buffer ``m ← β·m + c_best·Z_best`` and the
    step becomes ``θ -= lr·m``. Averages many (near-unbiased) probes over time, so the
    effective alignment with the true gradient grows ~√(1/(1-β)) — the cheapest way to
    sharpen the direction without any backward. Costs one extra buffer per parameter.
    Typical: 0.9 (effective ~10-step averaging)."""

    kronzo_hessian: bool = False
    """KronZO: enable the HiZOO diagonal-Hessian pre-conditioner. Uses the (free, in two_side
    mode) curvature ``S = (ℓ₊+ℓ₋−2ℓ)/2ε²`` to maintain a per-parameter ``Σ⁻¹`` and takes every
    perturbation/step along ``Σ^½ ⊙ z``. Requires ``two_side`` and no momentum / line search.
    Adds one O(d) float32 buffer per parameter."""

    kronzo_hessian_smooth: float = 1e-6
    """KronZO: EMA rate α for the diagonal-Hessian update (HiZOO Eq. 5). Only used with
    ``kronzo_hessian=True``."""

    kronzo_hessian_min: float = 1e-8
    """KronZO: lower clamp on the diagonal Hessian ``Σ⁻¹`` (keeps ``Σ^½`` finite). Only used
    with ``kronzo_hessian=True``."""

    kronzo_line_search_steps: int = 0
    """KronZO: after an accepted step, keep stepping along the same direction while the
    (same-batch) loss keeps dropping, up to this many extra steps (0 disables). Rides good
    directions and amortizes the expensive 1+3q probe sampling over several real moves
    (adaptive step length). Each extension is one extra forward; keep small (3-5) since the
    direction is chosen on a single minibatch."""

    kronzo_layer_sparsity: float = 0.0
    """KronZO + LeZO: fraction of parameter tensors ("layers") frozen each step (0 disables
    → plain KronZO; must be in [0, 1)). Implements the layer-wise sparsity of LeZO
    (arXiv:2410.09823) on top of KronZO's directional update: each step only the active
    subset is perturbed (A⊗B probes) and updated, while the rest stays put. The active set
    rotates deterministically with the step count — identical on every DDP rank without any
    communication — so over many steps all parameters are still trained (full-parameter
    fine-tuning). Two wins: (1) less perturbation/update tensor overhead per step (LeZO's
    original motivation), and (2) each probe lives in a lower-dimensional subspace, so the
    per-step ZO/FO alignment (~1/√d_active) improves. Typical: 0.5-0.75. Setting this on
    ``kron_zo`` is what the project calls "KronZO-LeZO"; the AGZO family has its own
    ``agzo_layer_sparsity``."""

    # --- HiZOO (Hessian-Informed Zeroth-Order Optimizer, ICLR 2025) ---

    hizoo_smooth_scale: float = 1e-6
    """HiZOO: EMA rate α for the diagonal-Hessian estimate (Eq. 5,
    ``Σ⁻¹ₜ = (1−α)Σ⁻¹ₜ₋₁ + α|diag(Σ'ₜ)|``). Larger α adapts the pre-conditioner
    faster but too large impedes convergence or causes gradient explosion (paper
    Fig. 6); the good regime is ~1e-4…1e-8. μ (finite-diff step) reuses ``zo_eps``."""

    hizoo_hessian_min: float = 1e-8
    """HiZOO: numerical floor on the diagonal Hessian estimate ``Σ⁻¹`` so the
    pre-conditioner ``Σ^{1/2} = 1/√Σ⁻¹`` cannot blow up. Not in the paper; a small
    safety clamp. The Hessian starts at 1 and stays O(1) with small α in practice."""

    # --- AGZO / AGZOKron / AGZOKronRL (Activation-Guided Zeroth-Order, arXiv:2601.17261) ---

    agzo_rank: int = 1
    """AGZO family: rank r of the activation subspace per linear layer (paper uses
    r=1 to minimize storage). The perturbation is Δ = R·Aᵀ with R ∈ R^{d_out×r} Gaussian
    and A ∈ R^{d_in×r} the top-r activation directions."""

    agzo_power_iters: int = 3
    """AGZO family: number of subspace/power-iteration steps K used to extract the
    activation basis A from the layer's activation Gram (paper default K=3)."""

    agzo_step_interval: int = 1
    """AGZO family: how often the activation subspace is re-extracted (1 = every step,
    from the current center-forward activations). >1 reuses the stored basis for that many
    steps to save the capture/power-iteration cost. For ``agzo_kron_rl`` keep it > 1 (10 in
    the shipped configs): the learned μ lives in the R-space paired with A, so a basis that
    changes every step makes μ accumulate against a shifting frame."""

    agzo_query_budget: int = 8
    """AGZOKron / AGZOKronRL: number of activation-guided probe directions q per step. Each
    probe reuses the shared per-layer activation basis A but a fresh random factor R. The
    step keeps the single best-decreasing probe (KronZO 'lucky-RGE'). Cost 1 + 3q forward
    passes (two_side). A single AGZO probe has large |cosine| but random sign; selecting the
    best of q converts that amplitude into an aligned descent step."""

    agzo_history_length: int = 10
    """AGZOKron / AGZOKronRL: sliding-window length h for the directional-update acceptance
    test — the best probe's candidate loss must beat the worst of the last h accepted
    losses, otherwise the step is skipped. 0 disables the test (always accept)."""

    agzo_update_direction: str = "best"
    """Which direction AGZO-Kron steps along once its q probes are scored.

    ``best`` (default, historical) keeps the single lowest-loss probe. ``qrge`` averages
    all q, i.e. plain multi-query RGE.

    Measured on three A2 runs at three learning rates: the selected probe's alignment with
    the true gradient is a stable 0.65x that of the q-average (cosine_selected 1.8-2.0e-4
    vs cosine_qrge 2.7-3.1e-4). Selecting on lowest loss discards about a third of the
    alignment the probes already carry. Averaging is also cheaper, because the per-probe
    candidate evaluations exist only to rank probes: 1+3q forwards becomes 1+2q+1.
    """

    agzo_layer_sparsity: float = 0.0
    """AGZO family: LeZO-style fraction of parameter tensors frozen each step (0
    disables; must be in [0, 1)). The active subset rotates deterministically with the
    step count (identical across DDP ranks), so all parameters are still trained over time
    while each step perturbs/updates — and captures activations for — only a
    ``(1 - agzo_layer_sparsity)`` fraction. Cuts per-step work and lowers the probe
    dimension (better per-step alignment). Setting this on ``agzo_kron`` is what the
    project calls "AGZO-Kron-LeZO", and on ``agzo_kron_rl`` — "AGZO-Kron-LeZO-RL".
    Typical: 0.5."""

    # --- AGZOKronRL (AGZO-Kron + learned mean on the R factor, "P1") ---

    agzo_rl_variance: float = 1.0
    """AGZOKronRL: standard deviation σ of the probe factor ``R ~ N(M, σ²)`` (ZO-RL
    convention — this is the std, not the variance). ``σ=1`` at ``M=0`` reproduces plain
    AGZO's ``R ~ N(0, I)``. Also the denominator in the ES μ-update (``1/(q·σ²)``)."""

    agzo_rl_lr_mu: float = 1e-4
    """AGZOKronRL: learning rate η_μ of the score-function update of the learned mean M.
    Keep small relative to the weight lr. Negative → fall back to ``learning_rate``."""

    agzo_rl_mu_return: str = "perturbed"
    """AGZOKronRL: which loss is the return f_i feeding the ES M-update.
      * ``perturbed`` — f_i = L(θ+εΔ_i) (directional, sign-aware; recommended);
      * ``candidate`` — f_i = L(θ−η c_i Δ_i) (symmetric in Δ_i ↦ −Δ_i)."""

    agzo_rl_mu_init: str = "zero"
    """AGZOKronRL: initial M when ``agzo_rl_mu_init_from_fo`` is false.
      * ``zero``   — M_0 = 0 (R starts isotropic, exactly AGZO; M drifts via ES);
      * ``random`` — M_0 = random unit matrix per layer."""

    agzo_rl_mu_init_from_fo: bool = False
    """AGZOKronRL: if true, seed M once from a first-order gradient projected onto the
    activation basis (``M = ∇_W·A``), so the first probe mean ≈ the true gradient
    direction. Runs one FO forward-backward on the first batch of a fresh run
    (``global_step == 1``); see ``Trainer._train_step_zero_order``. Overrides ``agzo_rl_mu_init``."""

    agzo_rl_mu_init_normalize: bool = False
    """AGZOKronRL: only used with ``agzo_rl_mu_init_from_fo=True``. If true, L2-normalize
    the global FO gradient before projecting it onto A; if false, use the raw gradient."""

    agzo_rl_update: str = "es"
    """AGZOKronRL: mean-update rule.
      * ``es``  — score-function / evolution-strategies gradient on M (default);
      * ``cem`` — Cross-Entropy Method: move M toward the mean R of the ``agzo_rl_elite_frac``
        lowest-loss probes (``η_μ = agzo_rl_lr_mu`` is the step toward the elite mean; 1.0 =
        full CEM refit). ``σ²`` scaling and ``fitness`` are ignored for CEM."""

    agzo_rl_fitness: str = "advantage"
    """AGZOKronRL (``update='es'`` only): per-probe weighting.
      * ``advantage`` — mean-subtracted returns ``(q f_i − Σ f_j)/(q−1)``;
      * ``rank`` — centered ranks in ``[−0.5, +0.5]`` (best probe → −0.5), scale-free and
        robust to outliers (NES / OpenAI-ES fitness shaping)."""

    agzo_rl_mirrored: bool = False
    """AGZOKronRL: draw the R noise in antithetic pairs ``R = M ± σζ`` (cheap variance
    reducer for the mean update). Needs a sign-aware return — with ``agzo_rl_mu_return=
    'candidate'`` the pairs cancel (a warning is emitted); ``perturbed`` is the sensible pick."""

    agzo_rl_elite_frac: float = 0.5
    """AGZOKronRL (``update='cem'`` only): fraction of the q probes kept as the elite set,
    ``k = max(1, round(q·elite_frac))``. Must be in (0, 1]."""

    # --- LDSD (ZO-LDSD family) ---

    ldsd_muon_newtonschulz_steps: int = 5
    """LDSDMuon: number of Newton-Schulz iterations for orthogonalising the 2-D gradient estimate."""

    # --- LDSDRl / LDSDRlAdaMM / LDSDRlSgd (ZO-RL family) ---

    ldsd_rl_k: int = 10
    """LDSDRl / LDSDRlAdaMM: number of candidate perturbation directions evaluated per step."""

    ldsd_rl_variance: float = 1e-3
    """LDSDRl / LDSDRlAdaMM: std of N(μ, variance·I) used to sample perturbation directions."""

    ldsd_rl_beta: float = 0.9
    """LDSDRl: momentum coefficient for gradient accumulator (grad_accum = β·accum + (1-β)·grad)."""

    ldsd_rl_params_ratio: float = 0.1
    """LDSDRl: fraction of trainable parameters perturbed per step (sparse exploration)."""

    ldsd_rl_sgd_momentum: float = 0.9
    """LDSDRlSgd: SGD momentum coefficient (0.0 = vanilla SGD without momentum)."""

    ldsd_rl_mu_init_from_fo: bool = False
    """LDSDRl / LDSDRlAdaMM / LDSDRlSgd: if true, seed μ_0 from a real first-order gradient
    (one FO backward pass on the first training batch) instead of the algorithm's default
    random-unit-vector (LDSDRl) or zero (LDSDRlAdaMM / LDSDRlSgd) initialization."""

    ldsd_rl_mu_init_normalize: bool = False
    """LDSDRl / LDSDRlAdaMM / LDSDRlSgd: only used when ``ldsd_rl_mu_init_from_fo=True``.
    If true, L2-normalize the FO gradient (global norm across all trainable params) before
    assigning it to μ_0; if false, use the raw FO gradient."""

    # --- LDSDRlKron (ZO-RL policy + KronZO directional selection) ---

    ldsd_rl_kron_query_budget: int = 10
    """LDSDRlKron: number of probe directions q evaluated per step (plays the role of ``k``
    in plain LDSDRl and of ``q`` in KronZO). Cost is 1 + 3q forward passes (two_side) or
    1 + 2q (one_side)."""

    ldsd_rl_kron_history_length: int = 10
    """LDSDRlKron: length h of the sliding loss window for KronZO's acceptance test. The step
    is applied only if the best candidate loss beats the worst loss in the window. 0 disables
    the test (always accept)."""

    ldsd_rl_kron_apply: str = "kron"
    """LDSDRlKron: how θ moves once KronZO has picked the best direction (seed, c_best).
      * ``kron``     — pure KronZO step ``θ -= α·c_best·z_best`` (no extra optimizer state);
      * ``sign_sgd`` — LDSDRl-style: grad_accum ← β·accum + (1-β)·(c_best·z_best), θ -= α·sign(accum);
      * ``sgd``      — LDSDRlSgd-style SGD (+ optional momentum) on grad = c_best·z_best;
      * ``adamm``    — LDSDRlAdaMM-style AMSGrad on grad = c_best·z_best.
    KronZO only selects the direction; this chooses how it is consumed."""

    ldsd_rl_kron_mu_return: str = "perturbed"
    """LDSDRlKron: which loss is used as the return f_i feeding the (unchanged) ES μ-update.
      * ``perturbed`` — f_i = L(θ+ε·z_i); identical to plain LDSDRl and gives μ a directional
        (sign-aware) signal. Recommended.
      * ``candidate`` — f_i = L(θ − α·c_i·z_i); the same signal KronZO selects on, but it is
        symmetric in z_i ↦ −z_i so it carries no sign information for μ."""

    ldsd_rl_kron_hessian: bool = False
    """LDSDRlKron: enable the HiZOO diagonal-Hessian pre-conditioner (same mechanism as
    ``kronzo_hessian``) on top of the ZO-RL sampler. Requires ``two_side`` and
    ``ldsd_rl_kron_apply='kron'`` (sign_sgd drops the scaling, adamm already normalizes).
    Adds one O(d) float32 buffer per parameter."""

    ldsd_rl_kron_hessian_smooth: float = 1e-6
    """LDSDRlKron: EMA rate α for the diagonal-Hessian update. Only used with
    ``ldsd_rl_kron_hessian=True``."""

    ldsd_rl_kron_hessian_min: float = 1e-8
    """LDSDRlKron: lower clamp on the diagonal Hessian ``Σ⁻¹``. Only used with
    ``ldsd_rl_kron_hessian=True``."""

    ldsd_rl_mu_init: str = "zero"
    """LDSDRlKron: initial μ_0 when ``ldsd_rl_mu_init_from_fo`` is false.
      * ``zero``   — μ_0 = 0 (isotropic N(0, σ²) probes at start; μ drifts via ES);
      * ``random`` — μ_0 = random unit vector (plain LDSDRl default).
    Orthogonal to ``ldsd_rl_mu_init_from_fo`` (FO-gradient init overrides this on step 1)."""

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "name") and new_config.name == "decoupled_lionw":
                new_config.name = "lionw"
                if hasattr(new_config, "eps"):
                    del new_config.eps

        return new_config


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"
    cosine_linear_envelope = "cosine_linear_envelope"
    constant_with_warmup = "constant_with_warmup"


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """

    warmup_min_lr: Optional[float] = None
    """
    The starting LR during the warmup period. If not set this defaults to 10% of
    the target LR.
    """


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class InstanceFilterConfig(BaseConfig):
    repetition_max_period: int = 13
    repetition_min_period: int = 1
    repetition_max_count: int = 32


@dataclass
class DataConfig(BaseConfig):
    paths: Optional[List[str]] = None
    memmap_dtype: str = "uint16"
    datasets: Optional[Dict[str, List[str]]] = None
    label_mask_paths: Optional[List[str]] = None
    pad_direction: PaddingDirection = PaddingDirection.right
    generate_attention_mask: bool = False
    generate_doc_lengths: bool = False
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0
    seed: Optional[int] = None
    instance_filter: Optional[InstanceFilterConfig] = None
    custom_dataset: Optional[CustomDatasetConfig] = None

    @property
    def effective_memmap_dtype(self):
        try:
            # getattr will check this is part of numpy module, while np.dtype will check
            # if this is a valid numpy dtype.
            np.dtype(dtype := getattr(np, self.memmap_dtype))
        except (AttributeError, TypeError) as e:
            raise TypeError(f"Value {self.memmap_dtype} is not a valid numpy type") from e
        return dtype


@dataclass
class CustomDatasetCollatorConfig(BaseConfig):
    input_id_field: str = "input_ids"  #: The field in the dataset items that contains the input token IDs.
    attention_mask_field: Optional[str] = None  #: The field in the dataset items that contains the attention mask.
    attention_bias_field: Optional[str] = None  #: The field in the dataset items that contains the attention bias.
    label_mask_field: Optional[str] = None  #: The field in the dataset items that contains the label mask.
    index_field: Optional[str] = None  #: The field in the dataset items that contains the index of the item.
    instance_mask_field: Optional[str] = None  #: The field in the dataset items that contains the instance mask.
    doc_lens_field: Optional[str] = None  #: The field in the dataset items that contains the document lengths.
    metadata_field: Optional[str] = None  #: The field in the dataset items that contains the metadata.


@dataclass
class CustomDatasetConfig(BaseConfig):
    name: str  #: The name of the custom dataset class or function that will be used to load the dataset.
    module: Optional[
        str
    ] = None  #: The module where the custom dataset class is defined. If not set, the module will be inferred from the class name.
    args: Optional[Dict[str, Any]] = None  #: The arguments to pass to the custom dataset class or function
    collate_fn: Optional[
        str
    ] = None  #: The name of the collate function to use for the custom dataset. Assumes the collate function is defined in the same module as the custom dataset class unless specified otherwise using the full object path.
    token_field: Optional[str] = None  #: The field in the dataset items that contains the tokenized text.
    collate_config: Optional[CustomDatasetCollatorConfig] = field(
        default_factory=CustomDatasetCollatorConfig
    )  #: The configuration for the collate function to use for the custom dataset.


class EvaluatorType(StrEnum):
    downstream = "downstream"
    lm = "lm"


@dataclass
class EvaluatorConfig(BaseConfig):
    label: str
    type: EvaluatorType = EvaluatorType.lm
    data: DataConfig = field(default_factory=DataConfig)
    device_eval_batch_size: Optional[int] = None
    subset_num_batches: Optional[int] = None


class TruncationDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    truncate_direction: TruncationDirection = TruncationDirection.right


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "ai2-llm"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = None
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    backend: str = "inductor"
    """
    The backend to use.
    """

    dynamic: Optional[bool] = None
    """
    From the torch docs:
    
    Use dynamic shape tracing. When this is True, we will up-front attempt to generate a kernel that is as dynamic
    as possible to avoid recompilations when sizes change. This may not always work as some
    operations/optimizations will force specialization; use TORCH_LOGS=dynamic to debug overspecialization. When
    this is False, we will NEVER generate dynamic kernels, we will always specialize. By default (None), we
    automatically detect if dynamism has occurred and compile a more dynamic kernel upon recompile.
    """


class DistributedStrategy(StrEnum):
    ddp = "ddp"
    """
    Wrap OLMo in torch.nn.parallel.DistributedDataParallel to train across ranks.
    """

    fsdp = "fsdp"
    """
    Wrap OLMo in torch.distributed.fsdp.FullyShardedDataParallel to train across ranks.
    """

    single = "single"
    """
    Train on a single device, i.e., do not distribute training. For development and debugging.
    """


class DDPGradSyncMode(StrEnum):
    batch = "batch"
    """
    Synchronize gradients after computation at each bucket only at the last micro-batch.
    This is slightly faster than gradient syncs across each micro-batch but will consume more memory.
    Can use this mode only when `find_unused_params` is set to False.
    """

    micro_batch = "micro_batch"
    """
    Synchronize gradients after computation at each bucket per micro-batch.
    This will be slightly slower than gradient sync at the last micro-batch, but will consume less memory.
    Can use this mode with both option of `find_unused_params` but specifically recommended to use with `find_unused_params`
    set to True, to prevent errors.
    """


@dataclass
class DDPConfig(BaseConfig):
    grad_sync_mode: DDPGradSyncMode = DDPGradSyncMode.batch
    """
    Gradient sync mode for DDP

    Note: When `find_unused_params` is set, set `grad_sync_mode` to `micro_batch` as different micro-batches might activate
    different parts of the model, ex- MOEs.
    """

    find_unused_params: bool = False
    """
    (from torch documentation)

    This mode allows running backward on a subgraph of the model, and DDP finds out which parameters
    are involved in the backward pass by traversing the autograd graph from the model output and marking
    all unused parameters as ready for reduction. Note that traversing the autograd graph introduces extra overheads,
    so applications should only set find_unused_parameters to True when necessary.
    """


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """


@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile`` or you want to track the parameter norm during training.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    wrapping_strategy: Optional[FSDPWrapStrategy] = None
    """
    The wrapping strategy to use. If ``None``, the default, the model is wrapped with a single top-level
    FSDP instance.
    """

    precision: Optional[FSDPPrecision] = FSDPPrecision.pure

    hybrid_sharding_num_model_replicas: Optional[int] = None
    """
    The number of model instances, when using a hybrid sharding strategy.
    If not ``None``, this must divide the total number of nodes. If ``None``, the default,
    a model instance is used per node (as determined by ``get_world_size() // get_local_world_size()``).
    PyTorch's default HSDP behavior matches this default behavior.
    """


@dataclass
class SingleGPUConfig(BaseConfig):
    device: str = "auto"
    """
    Device to run single-device training.
    """

    def get_device(self):
        if self.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif self.device == "mps" and not torch.backends.mps.is_available():
            raise OLMoConfigurationError("MPS not available.")
        elif self.device == "cuda" and not torch.cuda.is_available():
            raise OLMoConfigurationError("CUDA not available.")
        else:
            return torch.device(self.device)


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"
    olmo_core = "olmo_core"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    one_in_eight = "one_in_eight"
    """
    Checkpoint one in eight transformer layers.
    """

    two_in_three = "two_in_three"
    """
    Checkpoint two out of every three transformer layers.
    """

    three_in_four = "three_in_four"
    """
    Checkpoint three out of four of every transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """


@dataclass
class HybridFOConfig(BaseConfig):
    """FO parameter subset for the experiment-local FO + OracleProbe path."""

    param_patterns: List[str] = field(
        default_factory=lambda: [r"transformer\.wte\.weight", r"transformer\.ff_out\.weight"]
    )
    learning_rate: float = 3.0e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8


@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    The name of the run.
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    OLMo Model configuration.
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    hybrid_fo: Optional[HybridFOConfig] = None
    """Optional FO parameter subset for the experiment's OracleProbe hybrid."""

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data_root: Optional[str] = None
    """
    Root of the data directory, e.g. ``/data/users/vtrofimov/ZO/OLMo-Zero-Order-Optimization/data``.

    Data paths refer to it through ordinary OmegaConf interpolation, the same way
    ``${run_name}`` is used elsewhere::

        data_root: /data/users/vtrofimov/ZO/OLMo-Zero-Order-Optimization/data
        data:
          paths:
            - ${data_root}/dcml/subset-6B

    When the config leaves this unset it falls back to the ``OLMO_DATA_ROOT`` environment
    variable, so a machine can supply the root without the config being edited. Paths that
    do not mention ``${data_root}`` are unaffected.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When restarting, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When restarting, use this to fast-forward the dataloader beyond the last checkpoint.
    This can be useful when restarting due to a loss spike in order to skip the data that
    corresponded to the spike.
    """

    evaluators: List[EvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    remote_save_folder: Optional[str] = None
    """
    A folder in a cloud bucket to upload saved checkpoints to.
    """

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: Optional[int] = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save unsharded training state checkpoint.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """

    no_pre_train_checkpoint: bool = False
    """
    Skip saving pre-train checkpoint.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from. If not set, then training begins from scratch.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```

    If `try_load_latest_save` is set and saved checkpoints exist, then `load_path` will be overriden
    by the latest saved checkpoint.
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    try_load_latest_save: bool = False
    """
    If set, then training will be resumed from the latest checkpoint in the local save folder, falling
    back to the latest checkpoint in the remote save folder if none exists. If there are no checkpoints
    in the local and remote save folders, then checkpoint loading will fall back to `load_path`.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    reset_trainer_state: bool = False
    """
    When this is set we don't restore the trainer state from a checkpoint.
    """

    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    new_style_checkpoints: Optional[bool] = None
    """
    Deprecated. Use ``sharded_checkpointer`` instead.
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_batch_size: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    device_eval_batch_size: int = 16
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    eval_subset_num_batches: int = -1
    """
    The number of batches to use for downstream evaluation from each dataset.
    """

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    device_train_grad_accum: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``device_train_batch_size // device_train_microbatch_size``.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    distributed_strategy: Optional[DistributedStrategy] = DistributedStrategy.fsdp
    """
    Distributed strategy for OLMo model (eg. single GPU, DDP, FSDP).
    """

    fsdp: Optional[FSDPConfig] = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    ddp: Optional[DDPConfig] = None
    """
    DDP settings.
    """

    single: SingleGPUConfig = field(default_factory=lambda: SingleGPUConfig(device="auto"))
    """
    Single device settings for GPU/CPU/MPS. Defaults to auto-detect the best device.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0.
    """

    auxiliary_loss_multiplier: Optional[float] = 1e-4
    """
    Used with `softmax_auxiliary_loss`. PaLM uses 1e-4, Chameleon uses 1e-5.
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    """

    extra_steps_after_cancel: int = 10
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    early_stopping_factor: Optional[float] = None

    save_data_indices: bool = True
    """
    Save training data indices from each batch for each worker.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    """
    The activation checkpointing strategy to use.
    """

    fused_loss: Optional[bool] = None
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    hf_datasets_cache_dir: Optional[str] = None
    """
    Deprecated, HF datasets are now stored in `olmo_data.hf_datasets`.

    Path to cache directory of HF datasets saved with `datasets.save_to_disk`.
    """

    module_outputs_save_steps: Optional[List[int]] = None
    """
    Outputs of model submodules are saved during the provided steps. Submodule outputs
    can be compared using `scripts/compare_module_outputs.py`.
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> Optional[MixedPrecision]:
        if self.fsdp is not None:
            if self.fsdp.precision is None:
                return None
            elif self.fsdp.precision == FSDPPrecision.pure:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=self.autocast_precision,
                    buffer_dtype=self.autocast_precision,
                )
            elif self.fsdp.precision == FSDPPrecision.mixed:
                return MixedPrecision(
                    param_dtype=self.autocast_precision,
                    reduce_dtype=torch.float32,
                    buffer_dtype=self.autocast_precision,
                )
            else:
                raise NotImplementedError(f"{self.fsdp.precision}")
        else:
            raise ValueError("self.fsdp is None!")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    new_config.activation_checkpointing = ActivationCheckpointingStrategy.whole_layer

            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(new_config.optimizer)

        return new_config

    @classmethod
    def _pre_resolve(cls, config: DictConfig) -> None:
        """
        Settle ``data_root`` before OmegaConf resolves the ``${data_root}`` interpolations
        that data paths use: fall back to ``$OLMO_DATA_ROOT`` when the config leaves it
        unset, and fail loudly rather than letting an unset root resolve to the string
        ``"None"`` halfway through a path.
        """
        data_root = config.get("data_root") or os.environ.get(DATA_ROOT_ENV_VAR)
        if data_root:
            config.data_root = os.path.expanduser(str(data_root)).rstrip("/")
        elif "${data_root}" in om.to_yaml(config):
            raise OLMoConfigurationError(
                "Config uses ${data_root} in a path but never sets it. Set the 'data_root' "
                f"config field, pass --data_root=..., or export {DATA_ROOT_ENV_VAR}."
            )

    def _post_load(self, validate_paths: bool = True) -> None:
        """
        Expand directory-valued data paths, in the train data config and in every
        evaluator's data config. See :func:`expand_data_path`.
        """
        for data_config in [self.data] + [evaluator.data for evaluator in self.evaluators]:
            data_config.paths = expand_data_paths(data_config.paths, validate=validate_paths)
            data_config.label_mask_paths = expand_data_paths(data_config.label_mask_paths, validate=validate_paths)
            if data_config.datasets:
                data_config.datasets = {
                    label: cast(List[str], expand_data_paths(label_paths, validate=validate_paths))
                    for label, label_paths in data_config.datasets.items()
                }
