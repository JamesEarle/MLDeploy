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
compute_options = ["AdlaCompute", 
                   "AksCompute", 
                   "BatchAiCompute",
                   "ComputeTarget",
                   "DatabricksCompute",
                   "DataFactoryCompute",
                   "DsvmCompute",
                   "HDInsightCompute",
                   "RemoteCompute"]

parser = argparse.ArgumentParser()

# Required CLI arguments to deploy
parser.add_argument("--workspace-name", "-ws", type=str, dest="ws_name", help="Azure workspace to deploy to. Will create a workspace if not found.")
parser.add_argument("--experiment-name", "-ex", type=str, dest="exp_name", help="Workspace experiment to use. Will create an experiment if not found.")
parser.add_argument("--resource-group", "-rg", type=str, dest="rg", help="Resource group to use or create.")
parser.add_argument("--location", "-l", type=str, dest="location", help="Region for your resources to be deployed to.")
parser.add_argument("--compute-target", "-ct", type=str, dest="compute_target", default=None,
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

# Keep proper casing for compute option, regardless of what is entered (case insensitive arg)
compute_option = None
err = False

# Validate CLI args
for attr, value in args.__dict__.items():
    if(attr == "compute_target" and value != None):
        for co in compute_options: 
            if value.lower() == co.lower():
                compute_option = co
        # Not a valid selection
        if(compute_option == None):
            # Invalid compute selection, show list of available options
            print("Invalid compute target chosen. Please select one from the following list.\n{}".format(compute_options))

    if(value == None and attr != "compute_target"): # compute_target is optional arg
        # Want it to print all missing args, so call sys.exit after the loop
        print("Missing required argument: {}".format(attr))
        err = True
if err:
    sys.exit()

if(args.compute_target == None):
    print("Preparing for local training...")
else:
    # See if we can make this print nicely, instead of just what the user puts in
    print("Preparing for remote training, creating {} resource...".format(compute_option))

 
# Start by getting or creating the Azure workspace.
try:
    ws = Workspace.get(args.ws_name, subscription_id=sub_key)
    if(ws != None):
        print("Using workspace {}".format(args.ws_name))
except:
    print("Workspace not found, creating workspace {}".format(args.ws_name))
    ws = Workspace.create(name=args.ws_name,
                          subscription_id=sub_key,
                          resource_group=args.rg,
                          create_resource_group=True, # Change to false if you want to use a pre-existing resource group.
                          location=args.location)

if(compute_option == None): # does this use case really matter? Still need to create image and deploy
    # Train locally
    # Get the experiment or start a new one
    exp = Experiment(workspace=ws, name=args.exp_name)

    # Run model once, log results, and register in Azure.
    run = exp.start_logging()

    # Import your model. Ensure all required frameworks/packages are installed.
    import model

    # Add your own "run.log()" or "run.log_list()" calls in "model.train_and_save()"
    model_name, model_path = model.train_and_save(run) # local train

    # Register model on Azure
    model = run.register_model(model_name=model_name, model_path=model_path)
    print(model.name, model.id, model.version, sep = '\t')

    # Complete the run
    run.complete()

    # Print URL to show this experiment run in Azure portal.
    print(run.get_portal_url())
else:
    # Train remotely, get/create compute target
    print("remote")






# delete v
sys.exit()
# delete ^








# Instead of training and saving here, create the desired compute target 
# then create an estimator and submit the job to that compute target.
# https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml#create-an-estimator



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


