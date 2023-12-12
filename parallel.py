from azureml.core import ScriptRunConfig, Environment, Experiment
from azureml.core.workspace import Workspace

# Set up Azure ML Workspace
# Replace 'your_workspace_name', 'your_subscription_id', 'your_resource_group' with your Azure ML workspace details
workspace_name = 'caption'
subscription_id = '6ac6d14b-0753-40d6-9551-b2c077211327'
resource_group = 'simon-rg'
ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)

# Define the environment
curated_env_name = 'acpt-pytorch-2.0-cuda11.7'  # Choose an environment that suits your PyTorch version
# pytorch_env = Environment.get(workspace=ws, name=curated_env_name)

# Define the training script launch command
# '--nproc_per_node' should be less than or equal to the number of GPUs on the node
num_gpus_per_node = 8
training_script = 'preprocess.py'
launch_cmd = f"python -m torch.distributed.launch --nproc_per_node {num_gpus_per_node} --use_env {training_script}".split()

# Set the compute target
compute_target = 'simon1'  # Replace with your Azure ML compute target

# Configure the ScriptRunConfig
run_config = ScriptRunConfig(
    source_directory='./',  # Replace with the directory containing your training script
    command=launch_cmd,
    compute_target=compute_target,
    # environment=pytorch_env,
)

# Submit the experiment
experiment_name = 'caption-pre-1'  # Replace with your experiment name
run = Experiment(ws, experiment_name).submit(run_config)
print("Experiment submitted for execution.")