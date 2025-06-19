import Env.configuration as configuration
import Services.job_status as job_status
import Services.finetune_base_model as finetune_base_model


# Main
if __name__ == "__main__":

    #create training client
    client = finetune_base_model.CreateClient() 

    #upload training file, get ID
    training_file_id = finetune_base_model.upload_training_file(configuration.secrets.FEEDBACK_FILE) 
 
    job_id = finetune_base_model.upload_training_file.create_fine_tune_job(
            training_file_id= training_file_id, 
            client = client) #use IDs for training 
    
    #trigger function to Update Json Config. 
    job_status.CheckStatus(job_id)




   

