import os
import glob
import readline

# from azureml.core.compute_target import DsvmCompute

from azureml.core.authentication import InteractiveLoginAuthentication

# BatchAI for example, can use DsvmCompute for neural nets
# from azureml.core.compute import ComputeTarget, BatchAiCompute
# from azureml.core.compute_target import ComputeTargetException

# Tab completion code taken from this gist
# https://gist.github.com/iamatypeofwalrus/5637895

def path_completer(text, state):
    line = readline.get_line_buffer().split()
    return [x for x in glob.glob(text+"*")][state]

def location_list_completer(text, state):
    # Add the rest to list
    regions = ["centralus", "eastus", "eastus2", "westus", "westus2"]

    line = readline.get_line_buffer()

    if not line:
        return [loc + " " for loc in regions][state]
    else:
        return [loc + " " for loc in regions if loc.startswith(line)][state]

def service_list_completer(text, state):
    # Add the rest to list
    services = ["ACI", "AKS"]
    line = readline.get_line_buffer()

    if not line:
        return [loc + " " for loc in services][state]
    else:
        return [loc + " " for loc in services if loc.startswith(line)][state]

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

params["rg_name"] = input("Enter resource group name: ")
params["ws_name"] = input("Enter workspace name: ")

readline.set_completer(service_list_completer)

# need to do recursive validation similar to input path
params["service"] = input("Deploy to: ACI or AKS? ")

readline.set_completer(location_list_completer)

# need to do recursive validation similar to input path
params["location"] = input("Enter location: ")

model_name = params["model_file_path"].split("\\")[-1].split(".")[0]

# Start by getting or creating the Azure workspace.
from azureml.core import Workspace
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

from azureml.core.image import ContainerImage
image_config = ContainerImage.image_configuration(execution_script = params["score_file_path"],
                                                  runtime = "python",
                                                  conda_file = params["env_file_path"])

from azureml.core.webservice import Webservice
if params["service"] == "ACI":
    from azureml.core.webservice import AciWebservice
    service_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)  
    params["compute_target"] = None         
    
    # ACI, no compute target needed
    service_name = model_name + "-svc" # here for no, abstract into function to keep code DRY
    services = Webservice.deploy(deployment_config = service_config,
                                        image_config = image_config,
                                        model_paths = [params["model_file_path"]],
                                        name = service_name,
                                        workspace = ws)    
else:
    from azureml.core.webservice import AksWebservice
    from azureml.core.compute import AksCompute, ComputeTarget

    aks_name = "aml-aks-2323"

    try:
        # Look for existing AksCompute target
        aks_target = AksCompute(ws, aks_name)
    except:


        # allowed values for VM size
        # Standard_A10,Standard_A11,Standard_A2,Standard_A2_v2,Standard_A2m_v2,Standard_A3,Standard_A4,
        # Standard_A4_v2,Standard_A4m_v2,Standard_A5,Standard_A6,Standard_A7,Standard_A8,Standard_A8_v2,Standard_A8m_v2,S
        # tandard_A9,Standard_D11,Standard_D11_v2,Standard_D12,Standard_D12_v2,Standard_D13,Standard_D13_v2,Standard_D14,
        # Standard_D14_v2,Standard_D15_v2,Standard_D2,Standard_D2_v2,Standard_D3,Standard_D3_v2,Standard_D4,Standard_D4_v2,
        # Standard_D5_v2,Standard_DS11,Standard_DS11_v2,Standard_DS12,Standard_DS12_v2,Standard_DS13,Standard_DS13_v2,
        # Standard_DS14,Standard_DS14_v2,Standard_DS15_v2,Standard_DS2,Standard_DS2_v2,Standard_DS3,Standard_DS3_v2,
        # Standard_DS4,Standard_DS4_v2,Standard_DS5_v2,Standard_F16,Standard_F16s,Standard_F2,Standard_F2s,Standard_F4,
        # Standard_F4s,Standard_F8,Standard_F8s,Standard_G1,Standard_G2,Standard_NC6,Standard_NC12,Standard_NC24,
        # Standard_NC24r,Standard_NV6,Standard_NV12,Standard_NV24,Standard_B1s,Standard_B1ms,Standard_B2s,Standard_B2ms,
        # Standard_B4ms,Standard_B8ms
        
        # Creating new AksCompute target
        prov_config = AksCompute.provisioning_configuration(agent_count = 3, vm_size="Standard_DS12_v2")
        # prov_config = AksCompute.provisioning_configuration()

        aks_target = ComputeTarget.create(ws, name = aks_name, provisioning_configuration = prov_config)
        aks_target.wait_for_completion(show_output=True)
        print(aks_target.provisioning_state)
        print(aks_target.provisioning_errors)

    service_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
    # service_config = AksWebservice.deploy_configuration()

    service_name = model_name + "-svc"
    service = Webservice.deploy(deployment_config = service_config,
                                        deployment_target=aks_target,
                                        image_config = image_config,
                                        model_paths = [params["model_file_path"]],
                                        name = service_name,
                                        workspace = ws)

    service.wait_for_deployment(show_output = True)
    print(service.state)
    print(service.scoring_uri)