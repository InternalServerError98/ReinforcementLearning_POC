import openai
import os
from dotenv import load_dotenv
import updateConfig

# load all environment variables
load_dotenv()

#Open AI Config
openai.api_key = os.getenv("OPEN_AI_KEY")

HITL_DIR = os.getenv("FEEDBACK_FILE") #get the feedback file directory

#Upload json to Open AI
def upload_training_file(filepath: str) -> str:
    with open(filepath, "rb") as f:
        res = openai.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded training file. File ID: {res.id}")
        return res.id

#Create a new fine tuning Job
def create_fine_tune_job(training_file_id: str, 
                         model="gpt-4.1-2025-04-14") -> str:
    job = openai.fine_tuning.jobs.create(training_file=training_file_id,
                                         model=model, 
                                         hyperparameters={
                                               "n_epochs": 5,
                                                "batch_size": 1,
                                                "learning_rate_multiplier": 0.1
                                         })
    print(f"Fine-tune job started. Job ID: {job.id}")
    return job.id

# Main
if __name__ == "__main__":

    training_file_id = upload_training_file(HITL_DIR) #upload training file, get ID
 
    job_id = create_fine_tune_job(
            training_file_id= training_file_id) #use IDs for training 
    
    #trigger function to Update Json Config. 
    updateConfig.updateModel(job_id)

    #prune existing records.
    open(HITL_DIR, "w").close()

   

