from openai import AzureOpenAI, OpenAI
import Env.configuration as configuration

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
         


def GenerateResponse(messages):
    
    #Create the OpenAI client based on the environment
    client = CreateClient()

    #Get the model name based on the environment
    MODEL =  configuration.secrets.MODEL_DEPLOYMENT_NAME if configuration.secrets.TRAINING_ENVIRONMENT == 'azure'  else configuration.secrets.MODEL_NAME


    response = OpenAI.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "text"
        },
        temperature= configuration.secrets.MODEL_TEMPERATURE,
        max_completion_tokens=configuration.secrets.MODEL_MAX_COMPLETION_TOKENS,
        top_p=configuration.secrets.MODEL_TOP_P,
        frequency_penalty=configuration.secrets.MODEL_FREQUENCY_PENALTY,
        presence_penalty=configuration.secrets.MODEL_PRESENCE_PENALTY
    )


    return response