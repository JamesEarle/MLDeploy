import os
from azureml.core import Workspace, Experiment, Run, ScriptRunConfig, RunConfiguration
from sklearn.externals import joblib

# Import the model defined and trained in our train.py script.
# import train

ws_name = "mldeployworkspace"
# exp_name = dir_name = "keras-mnist" 
exp_sklearn_name = dir_name = "sklearn-mnist" 
sub_key = os.getenv("AZURE_SUBSCRIPTION")
resource_group = "MLDeploy"
location = "westus2"

# Get workspace
ws = Workspace.get(ws_name, subscription_id=sub_key)

# Get experiment
exp = Experiment(workspace=ws, name=exp_sklearn_name)
print(exp)

# model, metrics = train.build_train_model()
model, metrics = train.build_train_sklearn_model()

# Create a new run
config = ScriptRunConfig(source_directory='.', script='train.py', run_config=RunConfiguration())
run = exp.submit(config)

# Create an outputs directory
os.makedirs('outputs', exist_ok=True)

# model.save("./outputs/keras-mnist-model.h5") # Keras save model
joblib.dump(value=model, filename='./outputs/sklearn_mnist_model.pkl') # sklearn save model

# run.log_list("Metric labels: ", model.metrics_names) # keras metrics names
run.log("sklearn-mnist: accuracy =", metrics) # sklearn accuracy 
# run.log_list("Metrics values: ", metrics)

print("sklearn-mnist", metrics)