import openai
import process_data
import os

#Generate the jsonl file and obtain it's path
Training_Data = process_data.generateTrainingFile()
Validation_Data = process_data.generateValidationFile()

#Open AI Config
openai.api_key = os.getenv("OPEN_AI_KEY")

#Upload json to Open AI
def upload_training_file(filepath: str) -> str:
    with open(filepath, "rb") as f:
        res = openai.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded training file. File ID: {res.id}")
        return res.id

#Create a new fine tuning Job
def create_fine_tune_job(training_file_id: str, 
                         validation_file_id: str,
                         model="gpt-4.1-2025-04-14") -> str:
    job = openai.fine_tuning.jobs.create(training_file=training_file_id,
                                         validation_file=validation_file_id,
                                         model=model, 
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


    job_id = create_fine_tune_job(
            training_file_id= training_file_id, 
            validation_file_id=validation_file_id) #use IDs for training and validation

    print(f"Fine-tune Job ID: {job_id}") #pending fine tuning job

