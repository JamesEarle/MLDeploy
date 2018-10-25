# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-to-aci

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import azureml.services
from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication

ws_name = "MLDeploy"
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

global ws

if not workspaces:
    # Empty dictionary, no workspaces exist on this subscription
    print("Empty")
else:
    for ws in workspaces:
        print(ws)
        # check if more than one, if only one take immediately

# print(workspaces.keys[0])

# Workspace.get()
# ws = create_workspace(ws_name, sub_key, resource_group, location, False)

# List all workspaces in the given resource group
# Opens Edge for auth, only happens once. Access token is stored
# print(Workspace.list(sub_key, resource_group=resource_group))

# Workspace.get