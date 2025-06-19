import Services.finetune_base_model as finetune_base_model
import process_data
import time
import Services.job_status as job_status


#Generate the jsonl file and obtain it's path
Training_Data = process_data.generateTrainingFile()
Validation_Data = process_data.generateValidationFile()

# Main
if __name__ == "__main__":
    #create training client
    client = finetune_base_model.CreateClient()

    training_file_id = finetune_base_model.upload_training_file(Training_Data, client) #upload training file, get ID
    validation_file_id = finetune_base_model.upload_training_file(Validation_Data, client) #upload validation file, get ID

    print("Waiting for files to be uploaded...")
    time.sleep(30)  # wait 30 seconds after uploading files, this is a precautionary step to ensure files are ready


    job_id = finetune_base_model.create_fine_tune_job(
            training_file_id= training_file_id, 
            validation_file_id=validation_file_id, 
            client=client) #use IDs for training and validation

    print(f"Fine-tune Job ID: {job_id}") #pending fine tuning job, for base model

    #Check the status of the fine tuning job
    job_status.CheckStatus(job_id)  

