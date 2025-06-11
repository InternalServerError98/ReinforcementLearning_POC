import time
import openai
from pathlib import Path
import json

def updateModel(jobid: str):

    while True:
        job_status = openai.fine_tuning.jobs.retrieve(jobid)
        status = job_status["status"]
        print(f"Job {jobid} status: {status}")
        
        if status in ["succeeded", "failed"]:
            fine_tuned_model = job_status["fine_tuned_model"]
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
        config = {"current_model_id": "", "previous_models": []}

    # Update config
    previous_model = config.get("current_model_id", "")
    if previous_model:
        config.setdefault("previous_models", [])
        if previous_model not in config["previous_models"]:
            config["previous_models"].append(previous_model)

    config["current_model_id"] = fine_tuned_model

    # Save updated config
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated current fine tuned model to : {fine_tuned_model}")

        


