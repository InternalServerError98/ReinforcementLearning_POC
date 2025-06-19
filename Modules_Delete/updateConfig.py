import time
import openai
from openai import AzureOpenAI
from pathlib import Path
import json
from dotenv import load_dotenv

load_dotenv()  # ✅ This loads the .env file into the environment

def updateModel_Azure(jobid: str):
    while True:
        job_status = AzureOpenAI.fine_tuning.jobs.retrieve(jobid)
        #status = job_status["status"]
        status = job_status.status
        print(f"Azure fine tuning Job {jobid} status: {status}")
        
        if status in ["succeeded", "failed"]:
            #fine_tuned_model = job_status["fine_tuned_model"]
            fine_tuned_model = job_status.fine_tuned_model
            updateModelConfig(fine_tuned_model)
            break
        
        time.sleep(60)  # wait 60 seconds between polls


def updateModel(jobid: str):

    while True:
        job_status = openai.fine_tuning.jobs.retrieve(jobid)
       
        status = job_status.status
        print(f"Job {jobid} status: {status}")
        
        if status in ["succeeded", "failed"]:
            #fine_tuned_model = job_status["fine_tuned_model"]
            fine_tuned_model = job_status.fine_tuned_model
            updateModelConfig(fine_tuned_model)
            break
        
        time.sleep(60)  # wait 60 seconds between polls

    return

def updateModelConfig(fine_tuned_model : str):
        
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

        


