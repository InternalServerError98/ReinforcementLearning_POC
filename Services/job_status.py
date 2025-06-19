from openai import AzureOpenAI, OpenAI
import Env.configuration as configuration
import time
from pathlib import Path
import json


#Create client based on the environment
def CreateClient():
     
     if configuration.secrets.TRAINING_ENVIRONMENT == 'azure':
         return AzureOpenAI(
            api_key=configuration.secrets.AZURE_OPENAI_KEY,  
            api_version=configuration.secrets.AZURE_OPENAI_VERSION,
            azure_endpoint = configuration.secrets.AZURE_OPENAI_ENDPOINT
            )
     else:
         
         return OpenAI(
            api_key=configuration.secrets.OPEN_AI_KEY,  
            api_type=configuration.secrets.TRAINING_ENVIRONMENT
        )   
     

#Function that checks the status of the fine tuning job
def CheckStatus(job_id: str):

    #Create the OpenAI client based on the environment
    client = CreateClient()

    while True:
        #Get the job status
        job_status = client.fine_tuning.jobs.retrieve(job_id)
       
        status = job_status.status
        print(f"Job {job_id} status: {status}")
        
        if status in ["succeeded", "failed"]:
            fine_tuned_model = job_status.fine_tuned_model
            if(configuration.secrets.TRAINING_ENVIRONMENT == 'azure'):
                DeployModel(fine_tuned_model)
            else:
                UpdateModelConfig(fine_tuned_model)
            break
        
        time.sleep(60)  # wait 60 seconds between polls

    return



def DeployModel(fine_tuned_model):
    
    print(f"Model {fine_tuned_model} is ready, however auto deployment is not yet configured. Please deploy the model manually in Azure AI Foundary, and update the model_id in the model_config.json file.")
    return



def UpdateModelConfig(fine_tuned_model):
        
    CONFIG_PATH = Path("model_config.json")

    # Load existing config
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {"current_model_id": "", "model_history": []}

    # Update config
    previous_model = config.get("current_model_id", "")
    if previous_model:
        config.setdefault("model_history", [])
        if previous_model not in config["model_history"]:
            config["model_history"].append(previous_model)

    config["current_model_id"] = fine_tuned_model

    # Save updated config
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated current fine tuned model to : {fine_tuned_model}")