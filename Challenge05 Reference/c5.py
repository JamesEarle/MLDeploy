# challenge05.py

import os
import numpy as np

from keras.models import load_model

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import ContainerImage
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice

model_name = "keras-gear-cnn-0.8418.h5"
sub_key = os.getenv("AZURE_SUBSCRIPTION")

print("Checking score file... ", os.path.isfile("score.py"))
print("Checking env file... ", os.path.isfile("myenv.yml"))

location = "eastus2"
ws_name = "myworkspace"
resource_group = "myresourcegroup"

try:
    ws = Workspace.get(ws_name, subscription_id=sub_key)
    if(ws != None):
        print("Using workspace {}".format(ws_name))
except:
    print("Workspace not found, creating workspace {}".format(ws_name))
    ws = Workspace.create(name=ws_name,
                          subscription_id=sub_key,
                          resource_group=resource_group,
                          create_resource_group=True, # Change to false if you want to use a pre-existing resource group.
                          location=location)

print(ws.name)

print("Creating image config")

image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "Image for Keras CNN gear classification",
                                                  tags = {"data": "gear", "type": "classification"})

print("Creating ACI config")                                                  

aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {"data": "gear", "type": "classification"},
                                               description = 'Gear classification')                                                  

print("Creating ACI Webservice deployment")

service_name = 'my-gear-svc'
service = Webservice.deploy(deployment_config = aci_config,
                                image_config = image_config,
                                model_paths = [model_name],
                                name = service_name,
                                workspace = ws)

service.wait_for_deployment(show_output = True)

print(service.state)

print(service.scoring_uri)