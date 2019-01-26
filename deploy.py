import os
import sys
import argparse
import glob

import azureml.services
from azureml.core import Workspace
from azureml.core.image import ContainerImage
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication
# Should move imports around for efficiency instead of loading all up front, causing heavy start to script

# BatchAI for example, can use DsvmCompute for neural nets
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

# Tab completion code taken from this gist
# https://gist.github.com/iamatypeofwalrus/5637895
import readline

def path_completer(text, state):
    line = readline.get_line_buffer().split()
    return [x for x in glob.glob(text+"*")][state]

def list_completer(text, state):
    # Add the rest to list
    locations = ["centralus", "eastus", "eastus2", "westus", "westus2"]

    line = readline.get_line_buffer()

    if not line:
        return [loc + " " for loc in locations][state]
    else:
        return [loc + " " for loc in locations if loc.startswith(line)][state]

def validate_input_path(prompt): 
    input_path = input(prompt)
    if not os.path.isfile(input_path):
        print("Cannot find file {}\n".format(input_path))
        return validate_input_path(prompt)
    else:
        return input_path

readline.set_completer_delims('\t')
readline.parse_and_bind('tab: complete')
readline.set_completer(path_completer)

params = {}

params["model_file_path"] = validate_input_path("Path to model file: ")
params["score_file_path"] = validate_input_path("Path to score script: ")
params["env_file_path"] = validate_input_path("Path to conda environment file: ")

params["ws_name"] = input("Enter workspace name: ")
params["rg_name"] = input("Enter resource group name: ")

readline.set_completer(list_completer)

params["location"] = input("Enter location: ")

model_name = params["model_file_path"].split("\\")[-1].split(".")[0]

# Start by getting or creating the Azure workspace.
try:
    ws = Workspace.get(params["ws_name"], subscription_id=os.getenv("AZURE_SUBSCRIPTION"))
    if(ws != None):
        print("Using workspace {}".format(params["ws_name"]))
except:
    print("Workspace not found, creating workspace {}".format(params["ws_name"]))

    ws = Workspace.create(name=params["ws_name"],
                          subscription_id=os.getenv("AZURE_SUBSCRIPTION"),
                          resource_group=params["rg_name"],
                          create_resource_group=True, # Change to false if you want to use a pre-existing resource group.
                          location=params["location"])                   

image_config = ContainerImage.image_configuration(execution_script = params["score_file_path"],
                                                  runtime = "python",
                                                  conda_file = params["env_file_path"],
                                                  description = "Image for Keras CNN gear classification",
                                                  tags = {"data": "gear", "type": "classification"})

aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {"data": "gear", "type": "classification"},
                                               description = 'Gear classification')                       

service_name = model_name + "-svc"
service = Webservice.deploy(deployment_config = aci_config,
                                image_config = image_config,
                                model_paths = [params["model_file_path"]],
                                name = service_name,
                                workspace = ws)

service.wait_for_deployment(show_output = True)

print(service.state)

print(service.scoring_uri)