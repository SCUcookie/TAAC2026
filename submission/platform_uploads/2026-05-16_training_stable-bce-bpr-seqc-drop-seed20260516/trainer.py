"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import math
import contextlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
        pairwise_bpr_weight: float = 0.0,
        domain_dropout_probs: str = '',
        focal_alpha_mode: str = 'fixed',
        focal_alpha_start: float = 0.5,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.pairwise_bpr_weight: float = pairwise_bpr_weight
        self.domain_dropout_probs = self._parse_domain_dropout_probs(domain_dropout_probs)
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.log_every_n_steps: int = int(
            (train_config or {}).get("log_every_n_steps", 200)
        )
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.focal_alpha_mode = focal_alpha_mode
        self.focal_alpha_start = focal_alpha_start
        self._global_train_step = 0
        self._estimated_total_steps = max(1, len(train_loader) * max(1, num_epochs))

        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_shadow: Dict[str, torch.Tensor] = {}
        if use_ema:
            self.ema_shadow = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if p.requires_grad and p.is_floating_point()
            }
            logging.info(f"EMA enabled: {len(self.ema_shadow)} tensors, decay={ema_decay}")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"pairwise_bpr_weight={pairwise_bpr_weight}, "
                     f"domain_dropout_probs={self.domain_dropout_probs}, "
                     f"focal_alpha_mode={focal_alpha_mode}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _update_dense_lr(self) -> None:
        if self.warmup_steps <= 0 and self.min_lr_ratio >= 1.0:
            return
        step = max(1, self._global_train_step)
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            factor = step / float(self.warmup_steps)
        else:
            decay_steps = max(1, self._estimated_total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        lr = self.base_lr * factor
        for group in self.dense_optimizer.param_groups:
            group['lr'] = lr

    def _current_focal_alpha(self) -> float:
        if self.focal_alpha_mode == 'linear':
            progress = min(1.0, self._global_train_step / float(self._estimated_total_steps))
            return self.focal_alpha_start + (self.focal_alpha - self.focal_alpha_start) * progress
        return self.focal_alpha

    def _update_ema(self) -> None:
        if not self.use_ema:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.ema_shadow:
                    self.ema_shadow[name].mul_(self.ema_decay).add_(
                        p.detach(), alpha=1.0 - self.ema_decay)

    @contextlib.contextmanager
    def _ema_scope(self):
        if not self.use_ema:
            yield
            return
        backup = {}
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.ema_shadow:
                    backup[name] = p.detach().clone()
                    p.copy_(self.ema_shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if name in backup:
                        p.copy_(backup[name])

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        logging.info("TRAIN_START model=PCVRHyFormer epochs=%s train_steps_per_epoch=%s valid_steps=%s log_every_n_steps=%s",
                     self.num_epochs, len(self.train_loader), len(self.valid_loader),
                     self.log_every_n_steps)
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            logging.info("EPOCH_START epoch=%s", epoch)
            loss_sum = 0.0

            for step, batch in enumerate(self.train_loader, start=1):
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                if self.log_every_n_steps > 0 and step % self.log_every_n_steps == 0:
                    current_lr = self.dense_optimizer.param_groups[0]['lr']
                    logging.info(
                        "TRAIN_PROGRESS epoch=%s step=%s/%s total_step=%s avg_loss=%.6f last_loss=%.6f dense_lr=%.8g",
                        epoch, step, len(self.train_loader), total_step,
                        loss_sum / step, loss, current_lr,
                    )

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info("VALIDATION_START epoch=%s total_step=%s reason=step", epoch, total_step)
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info("VALIDATION_RESULT epoch=%s total_step=%s auc=%.12f logloss=%.12f reason=step",
                                 epoch, total_step, val_auc, val_logloss)

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    with self._ema_scope():
                        self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            avg_loss = loss_sum / len(self.train_loader)
            logging.info("EPOCH_TRAIN_RESULT epoch=%s total_step=%s avg_loss=%.6f",
                         epoch, total_step, avg_loss)

            logging.info("VALIDATION_START epoch=%s total_step=%s reason=epoch", epoch, total_step)
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info("VALIDATION_RESULT epoch=%s total_step=%s auc=%.12f logloss=%.12f reason=epoch",
                         epoch, total_step, val_auc, val_logloss)

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            with self._ema_scope():
                self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
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
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_inter_time_buckets[domain] = device_batch.get(
                f'{domain}_inter_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_time_deltas[domain] = device_batch.get(
                f'{domain}_time_delta',
                torch.zeros(B, L, dtype=torch.float32, device=self.device))
            seq_inter_time_deltas[domain] = device_batch.get(
                f'{domain}_inter_time_delta',
                torch.zeros(B, L, dtype=torch.float32, device=self.device))
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

    @staticmethod
    def _parse_domain_dropout_probs(raw: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for pair in (raw or '').split(','):
            pair = pair.strip()
            if not pair:
                continue
            k, v = pair.split(':')
            p = float(v.strip())
            if p > 0.0:
                out[k.strip()] = min(max(p, 0.0), 1.0)
        return out

    def _apply_domain_dropout(self, device_batch: Dict[str, Any]) -> None:
        if not self.domain_dropout_probs:
            return
        seq_domains = device_batch.get('_seq_domains', [])
        for domain in seq_domains:
            p = self.domain_dropout_probs.get(domain, 0.0)
            if p <= 0.0:
                continue
            mask = torch.rand(device_batch['label'].shape[0], device=device_batch['label'].device) < p
            if not mask.any():
                continue
            if domain in device_batch:
                device_batch[domain][mask] = 0
            len_key = f'{domain}_len'
            if len_key in device_batch:
                device_batch[len_key][mask] = 0
            for suffix in ('_time_bucket', '_inter_time_bucket', '_time_delta', '_inter_time_delta'):
                key = f'{domain}{suffix}'
                if key in device_batch:
                    device_batch[key][mask] = 0

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        self._global_train_step += 1
        self._update_dense_lr()
        device_batch = self._batch_to_device(batch)
        self._apply_domain_dropout(device_batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if self.loss_type == 'focal':
            if self.focal_alpha_mode == 'auto':
                pos_rate = label.mean().detach().clamp(min=1e-4, max=1 - 1e-4)
                alpha = float((1.0 - pos_rate).item())
            else:
                alpha = self._current_focal_alpha()
            loss = sigmoid_focal_loss(logits, label, alpha=alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        if self.pairwise_bpr_weight > 0.0:
            pairwise_loss = self._pairwise_bpr_loss(logits, label)
            loss = loss + self.pairwise_bpr_weight * pairwise_loss
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self._update_ema()

        return loss.item()

    @staticmethod
    def _pairwise_bpr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_logits = logits[labels > 0.5]
        neg_logits = logits[labels <= 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return logits.new_zeros(())
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        return F.softplus(-diff).mean()

    @staticmethod
    def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
        if len(probs) == 0 or len(np.unique(labels)) < 2:
            return 0.0
        return float(roc_auc_score(labels, probs))

    @staticmethod
    def _compute_gauc(labels: np.ndarray, probs: np.ndarray, users: np.ndarray) -> Tuple[float, int]:
        total_weight = 0
        weighted_auc = 0.0
        valid_groups = 0
        for uid in np.unique(users):
            mask = users == uid
            if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                continue
            auc = float(roc_auc_score(labels[mask], probs[mask]))
            w = int(mask.sum())
            weighted_auc += auc * w
            total_weight += w
            valid_groups += 1
        if total_weight == 0:
            return 0.0, 0
        return weighted_auc / total_weight, valid_groups

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        logging.info("VALIDATION_LOOP_START epoch=%s valid_steps=%s", epoch, len(self.valid_loader))
        self.model.eval()
        if not epoch:
            epoch = -1

        all_logits_list = []
        all_labels_list = []
        all_user_ids = []
        all_hist_lens = []

        with self._ema_scope():
            with torch.no_grad():
                for step, batch in enumerate(self.valid_loader, start=1):
                    logits, labels, hist_lens = self._evaluate_step(batch)
                    all_logits_list.append(logits.detach().cpu())
                    all_labels_list.append(labels.detach().cpu())
                    all_user_ids.extend(batch.get('user_id', []))
                    all_hist_lens.append(hist_lens.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()
        all_hist = torch.cat(all_hist_lens, dim=0).numpy() if all_hist_lens else np.array([])

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()
        users_np = np.asarray(all_user_ids)

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
            users_np = users_np[valid_mask]
            all_hist = all_hist[valid_mask]

        auc = self._safe_auc(labels_np, probs)
        gauc, gauc_groups = self._compute_gauc(labels_np, probs, users_np)

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        logging.info(f"Validation extra metrics | GAUC: {gauc}, GAUC_groups: {gauc_groups}")
        if self.writer:
            self.writer.add_scalar('GAUC/valid', gauc, self._global_train_step)
            self.writer.add_scalar('GAUC_groups/valid', gauc_groups, self._global_train_step)
        if len(all_hist) == len(probs) and len(probs) > 0:
            qs = np.quantile(all_hist, [0.25, 0.5, 0.75])
            bucket_ids = np.digitize(all_hist, qs, right=True)
            for bucket in range(4):
                mask = bucket_ids == bucket
                if mask.sum() == 0:
                    continue
                b_auc = self._safe_auc(labels_np[mask], probs[mask])
                logging.info(
                    f"Validation hist_bucket_{bucket} | rows={int(mask.sum())}, AUC={b_auc}")

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)
        hist_lens = torch.zeros_like(label, dtype=torch.float32)
        for domain in device_batch['_seq_domains']:
            hist_lens = hist_lens + device_batch[f'{domain}_len'].float()

        return logits, label, hist_lens
