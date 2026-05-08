● To install other packages, use conda or pip3.

For instance: to install a package like pandas (Note: pandas is pre-installed in the default environment; this is just an example).

 

# use conda
conda install -y pandas
 
# use pip3
pip3 install pandas
 

Model Training:
1.  Environment Variables
The platform provides directories required for training (e.g., training datasets, outputs, logs, etc.), and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache storage path (quota: 20GB). 

This variable is provided for both training and evaluation, allowing you to share files between these two stages.

TRAIN_DATA_PATH

Path to training datasets.

TRAIN_CKPT_PATH

Path for saving model checkpoints. 

TRAIN_TF_EVENTS_PATH

Path for TensorBoard event files.

In a shell script, you can read environment variables as follows:

 

${TRAIN_DATA_PATH}
 

In a Python script, you can read environment variables as follows：

 

import os
os.environ.get("TRAIN_DATA_PATH")
 

2.  Output Specifications
2.1. Model Output

a. Model weights must be saved to the directory specified by TRAIN_CKPT_PATH.

b. Iterative checkpoint files must be placed in a directory prefixed with global_step; otherwise, the platform will not recognize them. The directory name must not exceed 300 characters.

c. Directory names may only contain letters (a-z, A-Z), numbers (0-9), underscores (_), hyphens (-), equal signs (=), and periods (.).

 

To include additional parameter information in the directory name, append it after "global_step".

For example, a checkpoint from the 20th iteration with learning rate (lr=0.001), layer count (layer=2), head count (head=1), hidden size (hidden=50), and max sequence length (maxlen=200) should be saved under: "global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200"

 

2.2. TensorBoard Metrics

 

For PyTorch TensorBoard usage, refer to:https://docs.pytorch.org/docs/2.7/tensorboard.html#

 

As shown in the following code, you can read the "TRAIN_TF_EVENTS_PATH" environment variable to initialize a SummaryWriter. The platform only supports scalar metrics.

 

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
 

Model Evaluation:
1.Environment Variables
The platform provides directories needed during the evaluation process (such as evaluation datasets, outputs etc.) and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache path, quota 20GB.

This variable is provided for both training and evaluation, allowing you to share files between these two stages.

MODEL_OUTPUT_PATH

Model output path.

EVAL_DATA_PATH

Path to test data directory for inference.

EVAL_RESULT_PATH

Output path for intermediate results and "predictions.json".

EVAL_INFER_PATH

Directory for user-uploaded inference script files.

In a shell script, you can read environment variables as follows:

 

${EVAL_DATA_PATH}
 

In a Python script, you can read environment variables as follows：

 

import os
os.environ.get("EVAL_DATA_PATH")
 

2.  Output Specifications
The script must be strictly named "infer.py" and must contain a main() function that takes no arguments, otherwise the evaluation would fail!

The main() function must produce a "predictions.json" file and save it under the EVAL_RESULT_PATH directory.

 

● predictions.json Format

The JSON file must contain a predictions field, which is a mapping from user_id (string) to the predicted conversion probability (float, range 0–1).

Field

Data Format

Description

predictions

map<string, float>

A mapping where each key is a user_id (string) and each value is the model’s predicted probability that the user’s interaction is positive (i.e., a conversion).

 

● Example of predictions.json:

 

{
    "predictions": {
        "user_001": 0.8732,
        "user_002": 0.1245,
        "user_003": 0.5621
    }
}
 

 

Attention：Each key in the predictions map must be a valid user_id from the test dataset, and the corresponding value must be the model’s predicted conversion probability for that user’s interaction. Missing or extra user_ids may affect the final score!