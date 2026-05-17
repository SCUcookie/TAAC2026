"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
    'use_wide_dense_tokens': False,
    'wide_dense_threshold': 32,
    'dense_scalar_transform': 'none',
    'xdomain_dense_dim': 0,
    'use_xdomain_features': False,
    'use_inter_time_buckets': False,
    'use_continuous_time': False,
    'high_cardinality_mode': 'zero',
    'hash_num_buckets': 1048576,
    'hash_num_hashes': 2,
    'use_target_aware_attn': False,
    'use_cross_domain_attn': False,
    'seq_topk_mode': 'recent',
    'ffn_type': 'gelu',
    'norm_type': 'layernorm',
    'random_id_mask_prob': 0.0,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 64
_FALLBACK_NUM_WORKERS = 1
_DIAG_MAX_PROGRESS_LINES = 20
_FALLBACK_TTA_CROP_LENS = {
    'seq_a': 128,
    'seq_b': 128,
    'seq_c': 256,
    'seq_d': 256,
}
_FALLBACK_TTA_WEIGHT = 0.0
_FALLBACK_DOMAIN_ABLATION_WEIGHT = 0.04
_FALLBACK_DOMAIN_ABLATION_DOMAIN = 'seq_c'


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def _parse_primary_seq_fids(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not s:
        return out
    for pair in s.split(','):
        if not pair.strip():
            continue
        k, v = pair.split(':')
        out[k.strip()] = int(v.strip())
    return out


def _parse_tta_crop_lens(raw: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pair in raw.split(','):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(':')
        out[k.strip()] = int(v.strip())
    return out


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    if cfg.get('use_xdomain_features') and int(cfg.get('xdomain_dense_dim', 0)) <= 0:
        cfg['xdomain_dense_dim'] = 18
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        user_dense_schema_entries=dataset.user_dense_schema.entries,
        **model_cfg,
    )

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location='cpu')
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_inter_time_buckets: Dict[str, torch.Tensor] = {}
    seq_time_deltas: Dict[str, torch.Tensor] = {}
    seq_inter_time_deltas: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        seq_inter_time_buckets[domain] = device_batch.get(
            f'{domain}_inter_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))
        seq_time_deltas[domain] = device_batch.get(
            f'{domain}_time_delta',
            torch.zeros(B, L, dtype=torch.float32, device=device))
        seq_inter_time_deltas[domain] = device_batch.get(
            f'{domain}_inter_time_delta',
            torch.zeros(B, L, dtype=torch.float32, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        xdomain_dense_feats=device_batch.get('xdomain_dense_feats'),
        seq_inter_time_buckets=seq_inter_time_buckets,
        seq_time_deltas=seq_time_deltas,
        seq_inter_time_deltas=seq_inter_time_deltas,
    )


def _crop_model_input_recent(
    model_input: ModelInput,
    crop_lens: Dict[str, int],
) -> ModelInput:
    cropped_seq_data: Dict[str, torch.Tensor] = {}
    cropped_seq_lens: Dict[str, torch.Tensor] = {}
    cropped_seq_time_buckets: Dict[str, torch.Tensor] = {}
    cropped_seq_inter_time_buckets: Dict[str, torch.Tensor] = {}
    cropped_seq_time_deltas: Dict[str, torch.Tensor] = {}
    cropped_seq_inter_time_deltas: Dict[str, torch.Tensor] = {}

    for domain, seq in model_input.seq_data.items():
        B, S, L = seq.shape
        k = int(crop_lens.get(domain, L))
        k = max(1, min(k, L))
        lens = model_input.seq_lens[domain]
        time_buckets = model_input.seq_time_buckets[domain]
        device = seq.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        valid = pos < lens.unsqueeze(1)
        large = torch.full_like(time_buckets, 1_000_000)
        has_time = time_buckets > 0
        sort_key = torch.where(valid & has_time, time_buckets, large)
        fallback_key = torch.where(valid, pos + 100_000, large)
        sort_key = torch.where(has_time.any(dim=1, keepdim=True), sort_key, fallback_key)
        recent_idx = torch.argsort(sort_key, dim=1)[:, :k]
        recent_idx = torch.sort(recent_idx, dim=1).values

        gather_idx = recent_idx.unsqueeze(1).expand(B, S, k)
        cropped_seq = torch.gather(seq, dim=2, index=gather_idx)
        cropped_tb = torch.gather(time_buckets, dim=1, index=recent_idx)
        cropped_len = torch.clamp(lens, max=k)
        valid_new = torch.arange(k, device=device).unsqueeze(0) < cropped_len.unsqueeze(1)
        cropped_seq = cropped_seq * valid_new.unsqueeze(1).to(cropped_seq.dtype)
        cropped_tb = cropped_tb * valid_new.to(cropped_tb.dtype)

        cropped_seq_data[domain] = cropped_seq
        cropped_seq_lens[domain] = cropped_len
        cropped_seq_time_buckets[domain] = cropped_tb

        if model_input.seq_inter_time_buckets is not None:
            cropped_seq_inter_time_buckets[domain] = torch.gather(
                model_input.seq_inter_time_buckets[domain], dim=1, index=recent_idx)
            cropped_seq_inter_time_buckets[domain] = (
                cropped_seq_inter_time_buckets[domain] * valid_new.to(cropped_seq_inter_time_buckets[domain].dtype)
            )
        if model_input.seq_time_deltas is not None:
            cropped_seq_time_deltas[domain] = torch.gather(
                model_input.seq_time_deltas[domain], dim=1, index=recent_idx)
            cropped_seq_time_deltas[domain] = (
                cropped_seq_time_deltas[domain] * valid_new.to(cropped_seq_time_deltas[domain].dtype)
            )
        if model_input.seq_inter_time_deltas is not None:
            cropped_seq_inter_time_deltas[domain] = torch.gather(
                model_input.seq_inter_time_deltas[domain], dim=1, index=recent_idx)
            cropped_seq_inter_time_deltas[domain] = (
                cropped_seq_inter_time_deltas[domain] * valid_new.to(cropped_seq_inter_time_deltas[domain].dtype)
            )

    return ModelInput(
        user_int_feats=model_input.user_int_feats,
        item_int_feats=model_input.item_int_feats,
        user_dense_feats=model_input.user_dense_feats,
        item_dense_feats=model_input.item_dense_feats,
        seq_data=cropped_seq_data,
        seq_lens=cropped_seq_lens,
        seq_time_buckets=cropped_seq_time_buckets,
        xdomain_dense_feats=model_input.xdomain_dense_feats,
        seq_inter_time_buckets=(
            cropped_seq_inter_time_buckets if model_input.seq_inter_time_buckets is not None else None
        ),
        seq_time_deltas=(
            cropped_seq_time_deltas if model_input.seq_time_deltas is not None else None
        ),
        seq_inter_time_deltas=(
            cropped_seq_inter_time_deltas if model_input.seq_inter_time_deltas is not None else None
        ),
    )


def _ablate_sequence_domain(
    model_input: ModelInput,
    domain: str,
) -> ModelInput:
    if domain not in model_input.seq_data:
        return model_input

    seq_data = dict(model_input.seq_data)
    seq_lens = dict(model_input.seq_lens)
    seq_time_buckets = dict(model_input.seq_time_buckets)
    seq_inter_time_buckets = (
        dict(model_input.seq_inter_time_buckets)
        if model_input.seq_inter_time_buckets is not None else None
    )
    seq_time_deltas = (
        dict(model_input.seq_time_deltas)
        if model_input.seq_time_deltas is not None else None
    )
    seq_inter_time_deltas = (
        dict(model_input.seq_inter_time_deltas)
        if model_input.seq_inter_time_deltas is not None else None
    )

    seq_data[domain] = torch.zeros_like(seq_data[domain])
    seq_lens[domain] = torch.zeros_like(seq_lens[domain])
    seq_time_buckets[domain] = torch.zeros_like(seq_time_buckets[domain])
    if seq_inter_time_buckets is not None and domain in seq_inter_time_buckets:
        seq_inter_time_buckets[domain] = torch.zeros_like(seq_inter_time_buckets[domain])
    if seq_time_deltas is not None and domain in seq_time_deltas:
        seq_time_deltas[domain] = torch.zeros_like(seq_time_deltas[domain])
    if seq_inter_time_deltas is not None and domain in seq_inter_time_deltas:
        seq_inter_time_deltas[domain] = torch.zeros_like(seq_inter_time_deltas[domain])

    return ModelInput(
        user_int_feats=model_input.user_int_feats,
        item_int_feats=model_input.item_int_feats,
        user_dense_feats=model_input.user_dense_feats,
        item_dense_feats=model_input.item_dense_feats,
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        xdomain_dense_feats=model_input.xdomain_dense_feats,
        seq_inter_time_buckets=seq_inter_time_buckets,
        seq_time_deltas=seq_time_deltas,
        seq_inter_time_deltas=seq_inter_time_deltas,
    )


def _summarize_first_batch(batch: Dict[str, Any], batch_size: int) -> None:
    user_ids = batch.get('user_id', [])
    logging.info(
        "DIAG_FIRST_BATCH rows=%s unique_users=%s batch_keys=%s",
        len(user_ids),
        len(set(user_ids)) if user_ids else 0,
        sorted(str(k) for k in batch.keys()),
    )
    seq_domains = batch.get('_seq_domains', [])
    for domain in seq_domains:
        lens = batch.get(f'{domain}_len')
        if isinstance(lens, torch.Tensor) and lens.numel() > 0:
            lens_f = lens.float()
            nonzero_rate = float((lens > 0).float().mean().item())
            logging.info(
                "DIAG_SEQ domain=%s len_min=%.0f len_mean=%.4f len_p50=%.0f len_p90=%.0f len_max=%.0f nonzero_rate=%.4f cap=%s",
                domain,
                float(lens_f.min().item()),
                float(lens_f.mean().item()),
                float(torch.quantile(lens_f, 0.50).item()),
                float(torch.quantile(lens_f, 0.90).item()),
                float(lens_f.max().item()),
                nonzero_rate,
                batch.get(domain).shape[-1] if isinstance(batch.get(domain), torch.Tensor) else "NA",
            )
    for key in ("user_int_feats", "item_int_feats", "user_dense_feats", "item_dense_feats", "xdomain_dense_feats"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            finite_rate = float(torch.isfinite(value.float()).float().mean().item())
            logging.info(
                "DIAG_TENSOR key=%s shape=%s finite_rate=%.6f",
                key,
                tuple(value.shape),
                finite_rate,
            )


def _summarize_scores(scores: List[float], user_ids: List[str], elapsed_sec: float) -> None:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        logging.info("DIAG_SCORE empty=true")
        return
    quantiles = np.quantile(arr, [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    logging.info(
        "DIAG_SCORE count=%s unique_users=%s elapsed_sec=%.3f rows_per_sec=%.3f min=%.10f max=%.10f mean=%.10f std=%.10f p001=%.10f p01=%.10f p05=%.10f p50=%.10f p95=%.10f p99=%.10f p999=%.10f",
        int(arr.size),
        len(set(user_ids)),
        elapsed_sec,
        float(arr.size / elapsed_sec) if elapsed_sec > 0 else 0.0,
        float(arr.min()),
        float(arr.max()),
        float(arr.mean()),
        float(arr.std()),
        *[float(x) for x in quantiles],
    )
    buckets = np.linspace(0.0, 1.0, 11)
    hist, _ = np.histogram(arr, bins=buckets)
    logging.info("DIAG_SCORE_HIST bins=0.0:0.1:...:1.0 counts=%s", hist.tolist())


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")
    primary_seq_fids = _parse_primary_seq_fids(
        train_config.get('primary_seq_fids', ''))

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(os.environ.get('EVAL_BATCH_SIZE', str(_FALLBACK_BATCH_SIZE)))
    num_workers = int(os.environ.get('EVAL_NUM_WORKERS', str(_FALLBACK_NUM_WORKERS)))
    tta_enabled = os.environ.get('EVAL_RECENT_TTA', '1').lower() in {'1', 'true', 'yes'}
    tta_weight = float(os.environ.get('EVAL_RECENT_TTA_WEIGHT', str(_FALLBACK_TTA_WEIGHT)))
    tta_weight = min(max(tta_weight, 0.0), 0.20)
    domain_ablation_weight = float(os.environ.get(
        'EVAL_DOMAIN_ABLATION_WEIGHT',
        str(_FALLBACK_DOMAIN_ABLATION_WEIGHT),
    ))
    domain_ablation_weight = min(max(domain_ablation_weight, 0.0), 0.20)
    domain_ablation_domain = os.environ.get(
        'EVAL_DOMAIN_ABLATION_DOMAIN',
        _FALLBACK_DOMAIN_ABLATION_DOMAIN,
    )
    tta_crop_lens = _parse_tta_crop_lens(
        os.environ.get(
            'EVAL_RECENT_TTA_CROP_LENS',
            ','.join(f'{k}:{v}' for k, v in _FALLBACK_TTA_CROP_LENS.items()),
        )
    )
    logging.info(
        "DIAG_CONFIG batch_size=%s num_workers=%s seq_max_lens=%s primary_seq_fids=%s use_xdomain=%s use_inter_time=%s use_continuous_time=%s recent_tta=%s recent_tta_weight=%.4f recent_tta_crop_lens=%s domain_ablation_domain=%s domain_ablation_weight=%.4f",
        batch_size,
        num_workers,
        seq_max_lens,
        primary_seq_fids,
        bool(train_config.get('use_xdomain_features', False)),
        bool(train_config.get('use_inter_time_buckets', False)),
        bool(train_config.get('use_continuous_time', False)),
        tta_enabled,
        tta_weight,
        tta_crop_lens,
        domain_ablation_domain,
        domain_ablation_weight,
    )

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        use_xdomain_features=bool(train_config.get('use_xdomain_features', False)),
        primary_seq_fids=primary_seq_fids,
        use_inter_time_buckets=bool(train_config.get('use_inter_time_buckets', False)),
        use_continuous_time=bool(train_config.get('use_continuous_time', False)),
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")
    logging.info(
        "DIAG_DATASET rows=%s row_groups=%s seq_domains=%s",
        total_test_samples,
        len(getattr(test_dataset, '_rg_list', [])),
        getattr(test_dataset, 'seq_domains', []),
    )

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path} with CPU staging")
    load_model_state_strict(model, ckpt_path, device)
    del ckpt_path
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device)
    model.eval()
    logging.info("Model loaded successfully on %s", device)

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=1 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")
    start_time = time.time()
    progress_every = max(1, len(test_loader) // _DIAG_MAX_PROGRESS_LINES)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx == 0:
                _summarize_first_batch(batch, batch_size)
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            if tta_enabled and tta_weight > 0.0:
                recent_input = _crop_model_input_recent(model_input, tta_crop_lens)
                recent_logits, _ = model.predict(recent_input)
                logits = (1.0 - tta_weight) * logits + tta_weight * recent_logits
            if domain_ablation_weight > 0.0:
                ablated_input = _ablate_sequence_domain(model_input, domain_ablation_domain)
                ablated_logits, _ = model.predict(ablated_input)
                logits = ((1.0 - domain_ablation_weight) * logits
                          + domain_ablation_weight * ablated_logits)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % progress_every == 0:
                elapsed = time.time() - start_time
                rows = min((batch_idx + 1) * batch_size, total_test_samples)
                logging.info(
                    "DIAG_PROGRESS batch=%s/%s rows=%s elapsed_sec=%.3f rows_per_sec=%.3f",
                    batch_idx + 1,
                    len(test_loader),
                    rows,
                    elapsed,
                    rows / elapsed if elapsed > 0 else 0.0,
                )

    logging.info(f"Inference complete: {len(all_probs)} predictions")
    elapsed_total = time.time() - start_time
    _summarize_scores(all_probs, all_user_ids, elapsed_total)

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
