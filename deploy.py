# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-to-aci

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import azureml.services
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication

ws_name = "MLDeploy"
exp_name = "sklearn-mnist" 
sub_key = os.getenv("AZURE_SUBSCRIPTION")
resource_group = "MLDeploy"
location = "westus2"

# ila = new InteractiveLoginAuthentication()
# spa = ServicePrincipalAuthentication.get_authentication_header(sub_key)
# print(spa)

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

ws = Workspace.get(ws_name, subscription_id=sub_key)
print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

# Check for existing experiments
exp_names = ws.experiments()

print(exp_names)

# if not exp_names:
#     # none exist, create one
#     exp = Experiment(workspace=ws, name=exp_name)
# else:
#     # Some exist, show them
#     print(ws.experiments)

# print(exp)
# ws = create_workspace(ws_name, sub_key, resource_group, location, False)

# List all workspaces in the given resource group
# Opens Edge for auth, only happens once. Access token is stored
# print(Workspace.list(sub_key, resource_group=resource_group))

# Workspace.get