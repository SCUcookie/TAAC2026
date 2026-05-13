# TAAC2026 Project Progress Report

## 1. Project Goal

The current project is a TAAC2026 recommendation system pipeline for predicting the next ad a user is likely to interact with. The work is organized around a strict train/eval separation:

- training code is prepared in `baseline/training/`
- evaluation code is prepared in `baseline/evaluation/`
- platform upload archives are stored in `submission/platform_uploads/`
- immutable per-run code snapshots are stored in `submission/runs/`
- run-linked feedback and logs are stored in `feedbacks/`

The current operating rule is that every meaningful run must have a unique run id and the code snapshot, logs, and feedback must all point to that same run id.

## 2. Technical Underlying Principles

### 2.1 Data pipeline

The project does not treat the data as a simple dense table. It uses a schema-driven parquet pipeline:

- `schema.json` defines feature ids, offsets, lengths, and feature groups.
- `PCVRParquetDataset` reads parquet row groups as streaming batches.
- Each row group is converted into preallocated buffers for sparse ids, dense values, sequence features, and time features.
- Sequence domains such as `seq_a`, `seq_b`, `seq_c`, and `seq_d` are handled separately with configurable maximum lengths.

The design goal is to keep memory usage controlled while still supporting large-scale sparse and sequential features. This matters because the platform evaluation environment has a hard runtime budget and limited memory bandwidth.

### 2.2 Sparse and dense feature representation

The model works with two feature families:

- sparse categorical features, represented by embedding tables
- dense numeric features, represented by linear projections or feature transforms

Sparse features are the main signal source for user-item matching, while dense features provide side information such as counts, continuous signals, and multivariate metadata.

### 2.3 Sequence modeling

The model does not rely only on static feature joins. It also encodes user history sequences:

- each sequence domain is encoded independently
- padding masks prevent padded tokens from contributing to attention
- time gaps are bucketed into discrete bins so temporal patterns can be learned by embeddings rather than by raw continuous arithmetic

This approach turns sparse historical behavior into a structured temporal representation. It is especially useful when long sequences contain multiple modalities and when exact timestamps are less important than relative recency.

### 2.4 Ranking-oriented objective

The baseline training uses pointwise binary classification for click-like signals, but the current training direction also explores ranking-aware objectives.

- `bce` optimizes per-sample likelihood.
- `bce_pairwise` adds pairwise ranking pressure so positive samples should outrank negative samples.
- `focal` is available as a stable alternative when the runtime parser does not expose the pairwise path.

In ranking systems, pointwise loss alone can underweight the ordering structure that determines HitRate@10 and NDCG@10. Pairwise losses are meant to bridge that gap by directly rewarding score gaps between positive and negative examples.

### 2.5 Checkpoint self-containment

The checkpoint directory is treated as a self-contained artifact:

- `model.pt` stores the learned weights
- `schema.json` stores the feature layout
- `train_config.json` stores the training-time hyperparameters
- `ns_groups.json` stores the non-sequence feature grouping logic

This is important because evaluation must be able to load the exact feature mapping used during training. If any of these sidecar files are missing, the evaluator can silently drift from the training configuration or fail outright.

## 3. Model Framework

### 3.1 Core architecture

The active model family is the HyFormer-style ranking model used in the `baseline/training/` branch. Its main structure is:

1. feature embedding and projection
2. per-domain sequence encoding
3. cross-domain fusion through attention and token mixing
4. ranking head for final score prediction

### 3.2 Embedding layer

Sparse ids are mapped to embeddings and dense values are projected into the same latent space. The model typically uses:

- `d_model=64`
- `num_heads=4`
- `num_hyformer_blocks=2`
- `num_queries=2`
- `USER_NS_TOKENS=5`
- `ITEM_NS_TOKENS=2`

These values are a compact configuration that balances expressiveness and runtime cost.

### 3.3 Sequence encoder

The sequence encoder is configured as a transformer-style encoder in the current working runs. The sequence representation is built from:

- token embeddings
- position or recency signals
- time bucket embeddings
- padding-aware self-attention

The purpose is to let the model learn which historical events matter most, rather than simply averaging the full sequence.

### 3.4 Multi-sequence fusion

The model needs to combine several sequence domains and non-sequence token groups. The current design uses a HyFormer-like fusion mechanism:

- sequence summaries are extracted per domain
- query tokens attend over those summaries
- token groups are mixed through a lightweight transformer block

This gives the model a way to learn interactions across domains without flattening the entire history into one unstructured vector.

### 3.5 RankMixer and ranking head

The RankMixer tokenizer groups high-cardinality feature fields into token-like units. That helps the model avoid exploding the number of individual embeddings while still keeping field-level structure.

After feature fusion, a small MLP head outputs a single logit. The logit is then interpreted as a click or ranking score during evaluation.

### 3.6 Why this framework is suitable

The architecture is suitable for the task because it combines:

- sparse user-item matching
- sequential behavior modeling
- temporal recency bias
- cross-domain interaction modeling
- ranking-oriented scoring

That combination is more appropriate than a pure discriminator on static features, because the task depends heavily on order, recency, and feature interactions.

## 4. Current Workflow and File-Level Process

The project workflow is now explicitly file-linked.

### 4.1 Training flow

1. edit training code in `baseline/training/`
2. copy the exact submitted snapshot into `submission/runs/<run_id>/train/`
3. build the upload archive in `submission/platform_uploads/`
4. submit the archive to the platform
5. write the platform report into `feedbacks/platform_runs/<run_id>.md`
6. write raw logs into `feedbacks/logs_output/train/` or `feedbacks/logs_output/error/`
7. write the summary record into `feedbacks/records/<run_id>.md`

### 4.2 Evaluation flow

1. edit inference code in `baseline/evaluation/`
2. use the checkpoint created by the matching training run
3. copy the exact submitted snapshot into `submission/runs/<run_id>/eval/`
4. build the evaluation archive in `submission/platform_uploads/`
5. submit the archive to the platform
6. write the platform report into `feedbacks/platform_runs/<run_id>.md`
7. write raw logs into `feedbacks/logs_output/eval/` or `feedbacks/logs_output/error/`
8. write the summary record into `feedbacks/records/<run_id>.md`

### 4.3 Design rule

Training and evaluation are separated in code, but they are not disconnected:

- each evaluation run must point to one specific training checkpoint
- each training run may spawn more than one evaluation attempt
- the run id is the link that binds code, logs, and feedback together

## 5. Progress Analysis

### 5.1 What has been solved

The project has already solved the structural problem of messy experimentation.

- root-level Python files were removed from the top directory to reduce ambiguity
- the working split between `baseline/training/` and `baseline/evaluation/` is now established
- the reporting system now records each run separately by id
- training, evaluation, and error logs are split into dedicated folders

This is important because the user workflow depends on being able to look back at one run and immediately know which code, checkpoint, logs, and platform results belong together.

### 5.2 Training-side progress

The main recent training direction was the pairwise ranking experiment.

Observed outcome:

- the first `bce_pairwise` training submission failed early because the runtime parser accepted only `bce` and `focal`
- the failure was not a model crash; it was a runtime argument mismatch
- the launcher was then patched to fall back to `focal` when the runtime copy of `train.py` does not expose pairwise loss support

This tells us that the main issue on the training side is not architecture instability, but packaging and runtime compatibility.

### 5.3 Evaluation-side progress

The evaluation side has gone through several iterations on the original baseline checkpoint.

Key stages:

- early runs timed out because inference was too slow
- a BF16 no-merge path was found to be a stable reference evaluation
- the original baseline checkpoint produced a successful platform score of `0.813041`
- a later platform output in `temp.md` shows another evaluation that completed all `310000` predictions and wrote `predictions.json` and `status.json`

This means the evaluation pipeline is now functionally correct. The remaining work is not basic loading or serialization, but improvement in score and efficiency.

### 5.4 Benchmark position

The current practical benchmark is the best successful baseline evaluation result:

- baseline reference score: `0.813041`

Earlier records also show `0.813033` as the original baseline reference. The difference is small, but it matters because the project is now in a narrow optimization regime where incremental changes have to beat an already strong baseline.

### 5.5 Current bottlenecks

The main bottlenecks are now clear:

- training changes must be packaged so the platform runtime sees the intended parser and options
- evaluation cost is heavily affected by CPU-side batching and row-group conversion
- checkpoint compatibility and sidecar files are mandatory for reproducible inference
- score gains are small, so misleading improvements must be filtered out by local validation

## 6. Risk Analysis

### 6.1 Packaging risk

The platform does not always execute the exact same script state that exists locally. A change may exist in the local workspace but be absent or differently exposed in the runtime archive. The `bce_pairwise` parser failure is the best example of this risk.

### 6.2 Runtime risk

Evaluation can fail due to timeout even when the model is correct. In this project, throughput is affected by:

- row-group size
- CPU-side conversion
- sequence length caps
- batch merging behavior
- worker count and I/O overhead

That is why evaluation tuning is as important as model tuning.

### 6.3 Metrics risk

A successful run is not necessarily a better run. Some evaluation settings are slower but numerically identical, while others are faster but fail due to NaNs or overflow. The project therefore needs both:

- score validation
- runtime validation

### 6.4 Process risk

Without the new run-linked structure, it would be hard to know which log belongs to which submission. That would waste evaluation slots and make regression analysis unreliable. The new directory policy is the mitigation for that risk.

## 7. Recommended Next Steps

1. Keep the run-linked structure as the default experiment workflow.
2. Continue using `feedbacks/platform_runs/` and `feedbacks/logs_output/` as the primary source of truth for future analysis.
3. Use the latest successful baseline evaluation as the checkpoint for improvement comparisons.
4. For the next training experiment, prioritize compatibility and clear parser support before scaling complexity.
5. For the next evaluation experiment, keep a tight loop between local smoke checks and platform submissions so the limited daily evaluation budget is not wasted.

## 8. Detailed Model Notes

The full runtime flow and formula-level explanation has been moved to [docs/planning/2026-05-11_model_runtime_formulas.md](docs/planning/2026-05-11_model_runtime_formulas.md).

This report now keeps only the project-level workflow, progress, and risk analysis so it stays readable.

## 9. Short Conclusion

The project has moved from an unstructured codebase to a controlled experiment system. The training and evaluation pipelines are now separated, the run history is linked, and the baseline evaluation path is functionally stable. The main remaining challenge is not getting the pipeline to run, but improving score while preserving platform compatibility and runtime budget.