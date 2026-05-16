# Platform Run Record

## Basic Info

- Date: 2026-05-15
- Run type: train
- Submission directory: `submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515/`
- Intended change: focal loss on the proven `20260512` architecture

## Platform Result

Failed before training started.

Error:

`SyntaxError: invalid syntax` in `/home/taiji/dl/runtime/script/dataset.py`, line 2.

The platform-side `dataset.py` contained shell script text beginning with:

`SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`

## Decision

Treat this as a packaging/upload artifact mapping failure, not a focal-loss training failure. Do not reuse this package.

Retry with a clean package containing exactly the seven required files:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry.zip`

or the matching directory:

`submission/platform_uploads/2026-05-15_training_focal-hardfocus-seed20260515-cleanretry/`
