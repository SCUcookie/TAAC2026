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
import json
import logging
import math
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
    'legacy_user_dense_proj': False,
    'legacy_mixer_ffn_names': False,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 2048
_FALLBACK_NUM_WORKERS = 4
# Default evaluation window. The 256/256/256/256 run was very fast but lost
# AUC, while the full 512/512 long-domain window was too slow. Spend most of
# the available budget on the long domains and leave EVAL_SEQ_MAX_LENS as the
# explicit override for further sweeps.
_FALLBACK_EVAL_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:448'


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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {'1', 'true', 'yes', 'y', 'on'}


def _apply_eval_seq_max_lens_override(seq_max_lens: Dict[str, int]) -> Dict[str, int]:
    override = os.environ.get('EVAL_SEQ_MAX_LENS')
    if override:
        return _parse_seq_max_lens(override)
    cap = os.environ.get('EVAL_SEQ_LEN_CAP')
    if cap:
        cap_value = int(cap)
        return {domain: min(length, cap_value) for domain, length in seq_max_lens.items()}
    default_lens = _parse_seq_max_lens(_FALLBACK_EVAL_SEQ_MAX_LENS)
    return {
        domain: min(length, default_lens.get(domain, length))
        for domain, length in seq_max_lens.items()
    }


def _platform_event(name: str, payload: Dict[str, Any]) -> None:
    print(f"PLATFORM_{name} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def _file_status(path: str) -> Dict[str, Any]:
    return {
        'path': path,
        'exists': os.path.exists(path),
        'size_bytes': os.path.getsize(path) if os.path.exists(path) else 0,
    }


def _resolve_amp(device: str, use_amp: bool, requested_dtype: str) -> Tuple[bool, Optional[torch.dtype], str]:
    if not use_amp or device != 'cuda':
        return False, None, 'disabled'
    if requested_dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, 'bf16'
        _platform_event('AMP_FALLBACK', {
            'requested': 'bf16',
            'fallback': 'fp32',
            'reason': 'cuda_bf16_not_supported',
        })
        return False, None, 'fp32'
    return True, torch.float16, 'fp16'


def _sanitize_scores(scores: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    counts = {
        'nan_replaced': int(np.isnan(scores).sum()),
        'posinf_replaced': int(np.isposinf(scores).sum()),
        'neginf_replaced': int(np.isneginf(scores).sum()),
    }
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(scores, 0.0, 1.0), counts


def _score_stats(scores: List[float]) -> Dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {'count': 0}
    finite = arr[np.isfinite(arr)]
    stats: Dict[str, Any] = {
        'count': int(arr.size),
        'nan_count': int(np.isnan(arr).sum()),
        'inf_count': int(np.isinf(arr).sum()),
    }
    if finite.size == 0:
        return stats
    stats.update({
        'min': round(float(np.min(finite)), 8),
        'max': round(float(np.max(finite)), 8),
        'mean': round(float(np.mean(finite)), 8),
        'std': round(float(np.std(finite)), 8),
        'p01': round(float(np.quantile(finite, 0.01)), 8),
        'p50': round(float(np.quantile(finite, 0.50)), 8),
        'p99': round(float(np.quantile(finite, 0.99)), 8),
    })
    return stats


def _batch_row_count(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    return len(batch.get('user_id', []))


def _merge_cpu_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batches) == 1:
        return batches[0]
    merged: Dict[str, Any] = {}
    for key in batches[0].keys():
        first = batches[0][key]
        if isinstance(first, torch.Tensor):
            merged[key] = torch.cat([batch[key] for batch in batches], dim=0)
        elif key in {'user_id', 'item_id'}:
            values: List[Any] = []
            for batch in batches:
                values.extend(batch.get(key, []))
            merged[key] = values
        else:
            merged[key] = first
    return merged


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
    # Older checkpoints predate DenseFeatureTokenizer and the generic mixer
    # FFN module. Their train_config lacks these keys and stores weights under
    # user_dense_proj.* and blocks.*.mixer.fc{1,2}.*. Keep new checkpoints on
    # the new modules, but reconstruct old checkpoints exactly.
    if 'use_wide_dense_tokens' not in train_config and 'dense_scalar_transform' not in train_config:
        cfg['legacy_user_dense_proj'] = True
    if 'ffn_type' not in train_config and 'norm_type' not in train_config:
        cfg['legacy_mixer_ffn_names'] = True
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
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
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


def main() -> None:
    start_time = time.time()
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = _env_bool('EVAL_USE_AMP', True)
    amp_requested = os.environ.get('EVAL_AMP_DTYPE', 'bf16').lower()
    progress_every = int(os.environ.get('EVAL_PROGRESS_EVERY', '10'))
    merge_batches = _env_bool('EVAL_MERGE_BATCHES', False)

    _platform_event('INFER_START', {
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cwd': os.getcwd(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    })

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
    seq_max_lens = _apply_eval_seq_max_lens_override(_parse_seq_max_lens(sml_str))
    logging.info(f"seq_max_lens: {seq_max_lens}")
    primary_seq_fids = _parse_primary_seq_fids(
        train_config.get('primary_seq_fids', ''))

    # ---- Data loading: inference has independent throughput controls. ----
    batch_size = int(os.environ.get('EVAL_BATCH_SIZE', _FALLBACK_BATCH_SIZE))
    num_workers = int(os.environ.get('EVAL_NUM_WORKERS', _FALLBACK_NUM_WORKERS))

    _platform_event('CHECKPOINT_STATUS', {
        'checkpoint_dir': model_dir,
        'model_file': _file_status(os.path.join(model_dir, 'model.pt')),
        'schema_file': _file_status(schema_path),
        'train_config_file': _file_status(os.path.join(model_dir, 'train_config.json')),
    })
    _platform_event('INFER_CONFIG', {
        'checkpoint_dir': model_dir,
        'eval_data_dir': data_dir,
        'result_dir': result_dir,
        'schema_path': schema_path,
        'device': device,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'seq_max_lens': seq_max_lens,
        'use_amp': use_amp,
        'amp_dtype': amp_requested,
        'progress_every': progress_every,
        'merge_batches': merge_batches,
        'train_config': {
            'train_batch_size': train_config.get('batch_size'),
            'd_model': train_config.get('d_model'),
            'num_hyformer_blocks': train_config.get('num_hyformer_blocks'),
            'num_heads': train_config.get('num_heads'),
            'num_queries': train_config.get('num_queries'),
            'seq_encoder_type': train_config.get('seq_encoder_type'),
            'emb_skip_threshold': train_config.get('emb_skip_threshold'),
            'high_cardinality_mode': train_config.get('high_cardinality_mode'),
        },
    })

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        clip_vocab=True,
        use_xdomain_features=bool(train_config.get('use_xdomain_features', False)),
        primary_seq_fids=primary_seq_fids,
        use_inter_time_buckets=bool(train_config.get('use_inter_time_buckets', False)),
        use_continuous_time=bool(train_config.get('use_continuous_time', False)),
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")
    _platform_event('DATASET_CONFIG', {
        'batch_size': batch_size,
        'num_rows_estimate': total_test_samples,
        'num_row_groups': len(getattr(test_dataset, '_rg_list', [])),
        'seq_domains': test_dataset.seq_domains,
        'seq_max_lens': seq_max_lens,
        'user_int_features': len(test_dataset.user_int_schema.entries),
        'item_int_features': len(test_dataset.item_int_schema.entries),
        'user_dense_dim': test_dataset.user_dense_schema.total_dim,
        'item_dense_dim': test_dataset.item_dense_schema.total_dim,
    })

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
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")
    _platform_event('MODEL_READY', {
        'device': device,
        'num_ns': getattr(model, 'num_ns', None),
        'num_parameters': int(sum(p.numel() for p in model.parameters())),
    })

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")

    with torch.no_grad():
        amp_enabled, amp_dtype, amp_mode = _resolve_amp(device, use_amp, amp_requested)
        _platform_event('AMP_CONFIG', {
            'enabled': amp_enabled,
            'mode': amp_mode,
            'requested': amp_requested,
            'cuda_bf16_supported': torch.cuda.is_bf16_supported() if device == 'cuda' else False,
        })
        last_progress_time = start_time
        last_progress_predictions = 0
        timing_forward = 0.0
        timing_move = 0.0
        timing_post = 0.0
        timing_merge = 0.0
        pending_batches: List[Dict[str, Any]] = []
        pending_rows = 0
        forward_batches = 0
        row_group_batches = 0

        def run_batch(batch: Dict[str, Any], rows_in_forward: int) -> None:
            nonlocal amp_enabled, amp_dtype, amp_mode
            nonlocal last_progress_time, last_progress_predictions
            nonlocal timing_forward, timing_move, timing_post, timing_merge, forward_batches
            forward_batches += 1
            move_t0 = time.time()
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])
            move_t1 = time.time()

            autocast_kwargs = {'device_type': device, 'enabled': amp_enabled}
            if amp_dtype is not None:
                autocast_kwargs['dtype'] = amp_dtype
            with torch.amp.autocast(**autocast_kwargs):
                logits, _ = model.predict(model_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            forward_t1 = time.time()
            logits = logits.squeeze(-1)
            raw_probs = torch.sigmoid(logits).detach().float().cpu().numpy()
            if amp_enabled and not np.isfinite(raw_probs).all():
                _platform_event('AMP_FALLBACK', {
                    'requested': amp_mode,
                    'fallback': 'fp32',
                    'reason': 'nonfinite_scores',
                    'batch': forward_batches,
                    'nonfinite_scores': int((~np.isfinite(raw_probs)).sum()),
                })
                amp_enabled = False
                amp_dtype = None
                amp_mode = 'fp32_after_nonfinite'
                with torch.amp.autocast(device_type=device, enabled=False):
                    logits, _ = model.predict(model_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                raw_probs = torch.sigmoid(logits.squeeze(-1)).detach().float().cpu().numpy()
                forward_t1 = time.time()

            probs, sanitize_counts = _sanitize_scores(raw_probs.astype(np.float64))
            batch_probs = probs.astype(float).tolist()
            all_probs.extend(batch_probs)
            all_user_ids.extend(str(user_id) for user_id in user_ids)
            post_t1 = time.time()
            timing_move += move_t1 - move_t0
            timing_forward += forward_t1 - move_t1
            timing_post += post_t1 - forward_t1

            if forward_batches == 1:
                _platform_event('FIRST_BATCH', {
                    'batch_predictions': len(batch_probs),
                    'rows_in_forward': rows_in_forward,
                    'sample_user_ids': [str(x) for x in user_ids[:5]],
                    'sample_scores': [round(float(x), 8) for x in batch_probs[:5]],
                    'sanitize_counts': sanitize_counts,
                    'timing_sec': {
                        'merge': round(timing_merge, 4),
                        'move': round(move_t1 - move_t0, 4),
                        'forward': round(forward_t1 - move_t1, 4),
                        'post': round(post_t1 - forward_t1, 4),
                    },
                })
            elif any(sanitize_counts.values()):
                _platform_event('SCORE_SANITIZE', {
                    'batch': forward_batches,
                    'predictions': len(all_probs),
                    **sanitize_counts,
                })

            if progress_every > 0 and forward_batches % progress_every == 0:
                now = time.time()
                elapsed = max(now - start_time, 1e-6)
                since_last = max(now - last_progress_time, 1e-6)
                new_predictions = len(all_probs) - last_progress_predictions
                last_progress_time = now
                last_progress_predictions = len(all_probs)
                _platform_event('INFER_PROGRESS', {
                    'batches': forward_batches,
                    'row_group_batches': row_group_batches,
                    'predictions': len(all_probs),
                    'elapsed_sec': round(elapsed, 3),
                    'rows_per_sec': round(len(all_probs) / elapsed, 2),
                    'recent_rows_per_sec': round(new_predictions / since_last, 2),
                    'amp_mode': amp_mode,
                    'timing_avg_sec': {
                        'merge': round(timing_merge / forward_batches, 4),
                        'move': round(timing_move / forward_batches, 4),
                        'forward': round(timing_forward / forward_batches, 4),
                        'post': round(timing_post / forward_batches, 4),
                    },
                })

        def flush_pending() -> None:
            nonlocal pending_batches, pending_rows, timing_merge
            if not pending_batches:
                return
            merge_t0 = time.time()
            batch = _merge_cpu_batches(pending_batches) if len(pending_batches) > 1 else pending_batches[0]
            rows_in_forward = pending_rows
            pending_batches = []
            pending_rows = 0
            timing_merge += time.time() - merge_t0
            run_batch(batch, rows_in_forward)

        for row_group_batch in test_loader:
            row_group_batches += 1
            pending_batches.append(row_group_batch)
            pending_rows += _batch_row_count(row_group_batch)
            if not merge_batches or pending_rows >= batch_size:
                flush_pending()
        flush_pending()

    logging.info(f"Inference complete: {len(all_probs)} predictions")
    if len(all_user_ids) != len(all_probs):
        raise RuntimeError(f"Prediction count mismatch: {len(all_user_ids)} ids vs {len(all_probs)} scores")
    if len(set(all_user_ids)) != len(all_user_ids):
        raise RuntimeError("Duplicate user_id values detected in evaluation output")
    if any((not math.isfinite(float(score))) or float(score) < 0.0 for score in all_probs):
        raise RuntimeError("Invalid prediction score detected: scores must be finite and non-negative")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    scores_npy_path = os.path.join(result_dir, 'scores.npy')
    np.save(scores_npy_path, np.asarray(all_probs, dtype=np.float32))
    csv_path = os.path.join(result_dir, 'submission.csv')
    with open(csv_path, 'w') as f:
        f.write('user_id,score\n')
        for user_id, score in zip(all_user_ids, all_probs):
            f.write(f'{user_id},{float(score):.8f}\n')
    summary = {
        'num_predictions': len(all_probs),
        'num_unique_user_ids': len(set(all_user_ids)),
        'checkpoint_dir': model_dir,
        'elapsed_sec': round(time.time() - start_time, 3),
        'output_shape': 'predictions_dict_user_id_to_float',
        'result_json': output_path,
        'submission_csv': csv_path,
        'scores_npy': scores_npy_path,
        'result_files': {
            'json': _file_status(output_path),
            'csv': _file_status(csv_path),
            'scores': _file_status(scores_npy_path),
        },
        'score_stats': _score_stats(all_probs),
    }
    with open(os.path.join(result_dir, 'inference_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    _platform_event('INFER_SUMMARY', summary)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
