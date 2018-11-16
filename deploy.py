# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-to-aci
# 
# Sample run
# python deploy.py -ws mldeployworkspace -ex keras-mnist -rg MLDeploy -l westus2
import os
import sys
import argparse

import azureml.services
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication

# BatchAI for example, can use DsvmCompute for neural nets
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

sub_key = os.getenv("AZURE_SUBSCRIPTION")

parser = argparse.ArgumentParser()

# Required CLI arguments to deploy
parser.add_argument("--workspace-name", "-ws", type=str, dest="ws_name", help="Azure workspace to deploy to. Will create a workspace if not found.")
parser.add_argument("--experiment-name", "-ex", type=str, dest="exp_name", help="Workspace experiment to use. Will create an experiment if not found.")
parser.add_argument("--resource-group", "-rg", type=str, dest="rg", help="Resource group to use or create.")
parser.add_argument("--location", "-l", type=str, dest="location", help="Region for your resources to be deployed to.")
parser.add_argument("--compute-target", "-ct", type=str, dest="compute_target", 
                    help="Compute target to use or create. Options are: \
                        AdlaCompute, \
                        AksCompute, \
                        BatchAiCompute, \
                        ComputeTarget, \
                        DatabricksCompute, \
                        DataFactoryCompute, \
                        DsvmCompute, \
                        HDInsightCompute, \
                        RemoteCompute")
args = parser.parse_args()

err = False
for attr, value in args.__dict__.items():
    if(value == None):
        # Want it to print all missing args, so call sys.exit after the loop
        print("Missing required parameter: {}".format(attr))
        err = True
if err:
    sys.exit()

# Start by getting or creating the Azure workspace.
try:
    print("Using workspace {}".format(args.ws_name))
    ws = Workspace.get(args.ws_name, subscription_id=sub_key)
except:
    print("Workspace not found, creating workspace {}".format(args.ws_name))
    ws = Workspace.create(name=args.ws_name,
                          subscription_id=sub_key,
                          resource_group=args.rg,
                          create_resource_group=True, # Change to false if you want to use a pre-existing resource group.
                          location=args.location)

# Get the experiment or start a new one
exp = Experiment(workspace=ws, name=args.exp_name)

# Run model once, log results, and register in Azure.
run = exp.start_logging()

# Import your model. Ensure all required frameworks/packages are installed.
import model

# Instead of training and saving here, create the desired compute target 
# then create an estimator and submit the job to that compute target.
# https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml#create-an-estimator

# Add your own "run.log()" or "run.log_list()" calls in your model here.
model_name, model_path = model.train_and_save(run) # local train

model = run.register_model(model_name=model_name, model_path=model_path)
print(model.name, model.id, model.version, sep = '\t')

# Complete the run
run.complete()

# Print URL to show this experiment run in Azure portal.
print(run.get_portal_url())

# global compute_target

# # Search for or create compute resources
# try:
#     # should be = ComputeTarget? or = BatchAiCompute? Lint error on ComputeTarget
#     # _get?
#     compute_target = ComputeTarget(ws, name=cluster_name)

#     if(type(compute_target) is BatchAiCompute):
#         # found what we want, just use it
#         print("Resource {} found.".format(cluster_name))
#     else:
#         # name already exists on another resource
#         print("Resource '{0}' already exists is of type {1}.".format(cluster_name, type(compute_target)))
#         # print("A resource with name {} already exists, try another name.".format(cluster_name))
# except ComputeTargetException:
#     # create new resource because it wasn't found
#     print("Resource not found. Creating resource {}...".format(cluster_name))

#     # Can customize these parameters based on compute requirements.
#     # aka.ms/azureml-batchai-details for more info
#     compute_config = BatchAiCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",autoscale_enabled=True, cluster_min_nodes=0, cluster_max_nodes=4)

#     # Create the cluster
#     compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
#     compute_target.wait_for_completion(show_output=True)
#     print(compute_target.get_status())

# print(compute_target.get_status())

# cluster_name = "{}-cluster".format(args.exp_name)
# print("other stuff")