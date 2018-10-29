# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-to-aci

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import azureml.services
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication

from azureml.core.compute import ComputeTarget, BatchAiCompute # BatchAI for example, can use DsvmCompute for neural nets
from azureml.core.compute_target import ComputeTargetException

# ML resource strings
ws_name = "MLDeploy"
exp_name = "keras-mnist" 
sub_key = os.getenv("AZURE_SUBSCRIPTION")
resource_group = "MLDeploy"
location = "westus2"

# Compute resource strings, must be 2-16 characters, (a-zA-Z), (0-9), (-)
cluster_name = "mnist-cluster"

# Pass in a preexisting resource_group or flag to create one by changing createRG to be True
def create_workspace(name, sub, resource_group, location, createrg):
    return Workspace.create(name=name,
                      subscription_id=sub,
                      resource_group=resource_group,
                      create_resource_group=createrg,
                      location=location)

workspaces = Workspace.list(sub_key, resource_group=resource_group)

ws_names = []

if not workspaces:
    # Empty dictionary, no workspaces exist on this subscription
    print("Empty")
else:
    for idx, ws_name in enumerate(workspaces):
        ws_names.append(ws_name)

if len(ws_names) == 0:
    # create a workspace
    print(0)
elif len(ws_names) == 1:
    # take only option
    ws_name = ws_names[0]
else:
    print(2)
    # prompt for selection

# Get workspace and print info
ws = Workspace.get(ws_name, subscription_id=sub_key)
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

# Create experiment
exp = Experiment(workspace=ws, name=exp_name)
print(exp)

global compute_target

# Search for or create compute resources
try:
    # should be = ComputeTarget? or = BatchAiCompute? Lint error on ComputeTarget
    # _get?
    compute_target = ComputeTarget(ws, name=cluster_name)

    if(type(compute_target) is BatchAiCompute):
        # found what we want, just use it
        print("Resource {} found.".format(cluster_name))
    else:
        # name already exists on another resource
        print("Resource '{0}' already exists is of type {1}.".format(cluster_name, type(compute_target)))
        # print("A resource with name {} already exists, try another name.".format(cluster_name))
except ComputeTargetException:
    # create new resource because it wasn't found
    print("Resource not found. Creating resource {}...".format(cluster_name))

    # Can customize these parameters based on compute requirements.
    # aka.ms/azureml-batchai-details for more info
    compute_config = BatchAiCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",autoscale_enabled=True, cluster_min_nodes=0, cluster_max_nodes=4)

    # Create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)
    print(compute_target.get_status())


print(compute_target.get_status())
