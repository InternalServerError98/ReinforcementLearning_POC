import os
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from the appropriate .env file
env = os.getenv("ENV", "local")
load_dotenv(dotenv_path=f"Env/.env.{env}")

# Load config.json file
with open("model_config.json") as f:
    config_json = json.load(f)

#define a default model, in case the config.json does not have a current_model_id
default_model = 'gpt-4.1-2025-04-14'

@dataclass
class Config:
    #Add env variables
    ENV: str = field(default_factory=lambda: os.getenv("ENV", "local"))
    TRAINING_ENVIRONMENT: str = field(default_factory=lambda: os.getenv("TRAINING_ENVIRONMENT", "openai"))
    OPEN_AI_KEY: str = field(default_factory=lambda: os.getenv("OPEN_AI_KEY", ""))
    AZURE_OPENAI_ENDPOINT: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    AZURE_OPENAI_KEY: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_KEY", ""))
    AZURE_OPENAI_VERSION: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_VERSION", "2024-10-21"))
    FEEDBACK_FILE: str = field(default_factory=lambda: os.getenv("FEEDBACK_FILE", "Feedback/reinforce_data.jsonl"))
    FEEDBACK_URL: str = field(default_factory=lambda: os.getenv("FEEDBACK_URL", "http://localhost:8080/feedback"))
    INPUT_FILE_PATH: str = field(default_factory=lambda: os.getenv("INPUT_FILE_PATH", "Data/Train/Input"))
    OUTPUT_FILE_PATH: str = field(default_factory=lambda: os.getenv("OUTPUT_FILE_PATH", "Data/Train/Output"))
    VALIDATION_INPUT_PATH: str = field(default_factory=lambda: os.getenv("VALIDATION_INPUT_PATH", "Data/Train/Validation/Input"))
    VALIDATION_OUTPUT_PATH: str = field(default_factory=lambda: os.getenv("VALIDATION_OUTPUT_PATH", "Data/Train/Validation/Output"))
    
    INPUT_PATH: str = field(default_factory=lambda: os.getenv("TRAINING_JSONL", "training_data.jsonl"))
    OUTUT_PATH: str = field(default_factory=lambda: os.getenv("VALIDATION_JSONL", "validation_data.jsonl"))
    
    LEARNING_RATE_TRAIN: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE_TRAIN", 0.0001)))
    BATCH_SIZE_TRAIN: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE_TRAIN", 4)))
    EPOCHS_TRAIN: int = field(default_factory=lambda: int(os.getenv("EPOCHS_TRAIN", 6)))

    #Add Model Config Variables
    MODEL_NAME: str = field(default_factory=lambda: config_json.get("current_model_id", default_model))
    SYSTEM_PROMPT: str = field(default_factory=lambda: config_json.get("system_prompt", "You are a senior prompt engineer. You convert designs to prompts that help LLMs translate them into code."))
    MESSAGE_HEADER  = config_json.get("prompt_header")


    MODEL_TEMPERATURE: int = field(default_factory=lambda: config_json.get("temperature", 1))
    MODEL_MAX_COMPLETION_TOKENS: int = field(default_factory=lambda: config_json.get("max_completion_tokens", 2048))
    MODEL_TOP_P: int = field(default_factory=lambda: config_json.get("top_p", 1))
    MODEL_FREQUENCY_PENALTY: int = field(default_factory=lambda: config_json.get("frequency_penalty", 0))
    MODEL_PRESENCE_PENALTY: int = field(default_factory=lambda: config_json.get("presence_penalty", 0))
    BASE_MODEL = field(default_factory=lambda: config_json.get("base_model", "gpt-4.1-2025-04-14"))
    BASE_MODEL_SEED = field(default_factory=lambda: config_json.get("base_model_seed", 42))


secrets = Config()

