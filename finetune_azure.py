from openai import AzureOpenAI
import process_data
import os
import updateConfig
from dotenv import load_dotenv
import time

load_dotenv()  # ✅ This loads the .env file into the environment

#Generate the jsonl file and obtain it's path
Training_Data = process_data.generateTrainingFile()
Validation_Data = process_data.generateValidationFile()

#create training client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

#Upload json to Open AI
def upload_training_file(filepath: str) -> str:
    with open(filepath, "rb") as f:
        res = client.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded training file. File ID: {res.id}")
        return res.id

#Create a new fine tuning Job
def create_fine_tune_job(training_file_id: str, 
                         validation_file_id: str,
                         model="gpt-4.1-2025-04-14") -> str:
    job = client.fine_tuning.jobs.create(training_file=training_file_id,
                                         validation_file=validation_file_id,
                                         model=model, 
                                         seed=42,
                                         hyperparameters={
                                               "n_epochs": 6,
                                                "batch_size": 1,
                                                "learning_rate_multiplier": 0.1
                                         })
    print(f"Fine-tune job started. Job ID: {job.id}")
    return job.id

# Main
if __name__ == "__main__":

    training_file_id = upload_training_file(Training_Data) #upload training file, get ID
    validation_file_id = upload_training_file(Validation_Data) #upload validation file, get ID


    print("Waiting for files to be uploaded...")
    time.sleep(60)  # wait 60 seconds after uploading files


    job_id = create_fine_tune_job(
            training_file_id= training_file_id, 
            validation_file_id=validation_file_id) #use IDs for training and validation

    print(f"Fine-tune Job ID: {job_id}") #pending fine tuning job
    

