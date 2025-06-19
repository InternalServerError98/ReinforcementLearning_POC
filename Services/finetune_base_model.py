from openai import AzureOpenAI, OpenAI
import Env.configuration as configuration
import process_data


#Generate the jsonl file and obtain it's path
Training_Data = process_data.generateTrainingFile()
Validation_Data = process_data.generateValidationFile()


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
         
#Upload json to Open AI
def upload_training_file(filepath: str, client) -> str:
    with open(filepath, "rb") as f:
        res = client.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded training file. File ID: {res.id}")
        return res.id
    

#Create a new fine tuning Job
def create_fine_tune_job(training_file_id: str, 
                         validation_file_id: str, 
                         client) -> str:
    job = client.fine_tuning.jobs.create(training_file=training_file_id,
                                         validation_file=validation_file_id,
                                         model=configuration.secrets.BASE_MODEL, 
                                         seed=configuration.secrets.BASE_MODEL_SEED,
                                         hyperparameters={
                                               "n_epochs": configuration.secrets.EPOCHS_TRAIN,
                                                "batch_size": configuration.secrets.BATCH_SIZE_TRAIN,
                                                "learning_rate_multiplier":  configuration.secrets.LEARNING_RATE_TRAIN
                                         })
    print(f"Fine-tune job started. Job ID: {job.id}")
    return job.id