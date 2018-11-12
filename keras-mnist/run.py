import os
from azureml.core import Workspace, Experiment, Run, ScriptRunConfig, RunConfiguration

# Import the model defined and trained in our train.py script.
import train

ws_name = "mldeployworkspace"
exp_name = dir_name = "keras-mnist" 
sub_key = os.getenv("AZURE_SUBSCRIPTION")
resource_group = "MLDeploy"
location = "westus2"

# Get workspace
ws = Workspace.get(ws_name, subscription_id=sub_key)

# Get experiment
exp = Experiment(workspace=ws, name=exp_name)
print(exp)

model, metrics = train.build_train_model()

# Create a new run
config = ScriptRunConfig(source_directory='.', script='train.py', run_config=RunConfiguration())
run = exp.submit(config)

# Create an outputs directory
os.makedirs('outputs', exist_ok=True)
model.save("./outputs/keras-mnist-model.h5")

run.log_list("Metric labels: ", model.metrics_names)
run.log_list("Metrics values: ", metrics)

print(model.name, metrics)