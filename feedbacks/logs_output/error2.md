ModuleNotFoundError: No module named 'utils'
2026-05-08 15:47:24.245
from utils import create_logger
2026-05-08 15:47:24.245
File "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/71023/infer/infer.py", line 29, in <module>
2026-05-08 15:47:24.244
from infer import main as infer
2026-05-08 15:47:24.244
File "/workspace/eval.py", line 23, in <module>
2026-05-08 15:47:24.244
Traceback (most recent call last):
2026-05-08 15:47:24.244
2026-05-08 15:47:24.244
During handling of the above exception, another exception occurred:
2026-05-08 15:47:24.244
2026-05-08 15:47:24.244
ImportError: attempted relative import with no known parent package
2026-05-08 15:47:24.244
from .dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
2026-05-08 15:47:24.244
File "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/71023/infer/infer.py", line 23, in <module>
2026-05-08 15:47:24.243
Traceback (most recent call last):
2026-05-08 15:47:24.243
====== Inferring ======
2026-05-08 15:47:23.015
=================================
2026-05-08 15:47:22.622
Working Dir: /workspace
2026-05-08 15:47:22.622
Environment: competition
2026-05-08 15:47:22.622
GPU Count: 1
2026-05-08 15:47:22.622
GPU Available: True
2026-05-08 15:47:21.239
PyTorch: 2.7.1+cu126
2026-05-08 15:47:19.889
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-08 15:47:17.712
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-08 15:47:17.712
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-08 15:47:17.712
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-08 15:47:17.712
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-08 15:47:17.709
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-08 15:47:17.709
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-08 15:47:17.709
Python: Python 3.10.20
2026-05-08 15:47:17.358
CUDA: 12.6.77
2026-05-08 15:47:17.353
=== Competition Environment Ready ===
2026-05-08 15:47:17.343
Complete setting network policy rules.
2026-05-08 15:47:13.811
Complete setting taiji user.
2026-05-08 15:47:13.720