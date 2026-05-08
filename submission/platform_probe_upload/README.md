Upload only the file `infer.py` from this directory, or upload this directory
as a flat bundle only if the platform preserves `infer.py` at the root of the
workspace.

The platform error in `feedbacks/logs_output/error1.md`:

- `ModuleNotFoundError: No module named 'infer'`

means the platform did not see a root-level `infer.py` at import time.

So the safe upload method is:

1. Open this directory.
2. Select the file `infer.py`.
3. Upload that file as the evaluation script.
