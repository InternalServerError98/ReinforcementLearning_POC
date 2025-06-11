import os
import base64
import json
from dotenv import load_dotenv
import random
from striprtf.striprtf import rtf_to_text


# load all environment variables
load_dotenv()

TRAINING_INPUT_DIR = os.getenv("INPUT_FILE_PATH")
TRAINING_OUTPUT_DIR = os.getenv("OUTPUT_FILE_PATH")

VALIDATION_INPUT_DIR = os.getenv("VALIDATION_INPUT_PATH")
VALIDATION_OUTPUT_DIR = os.getenv("VALIDATION_OUTPUT_PATH")

TRAINING_JSONL_PATH = 'training_data.jsonl'
VALIDATION_JSONL_PATH = 'validation_data.jsonl'


USER_PROMPT_VARIANTS = [
    "Describe the UI shown in the image below.",
    "What UI components do you see in this screenshot?",
    "Give me a breakdown of this screen.",
    "Generate a prompt for this design", 
    "Help me generating a prompt for this design",
    "What are the components of this screen",
    "Generate a prompt for this UI", 
    "Help me build this UI", 
    "Help me build a prompt for this UI"
]

SYSTEM_PROMPT = "You are a lead prompt Engineer. You are trying to generate prompts from the UI images presented to you. Your job is to describe the image and all its funcions/designs in a way that LLMs or other models can develop seamless UI components from it."

# Creating the Jsonl file to load the data. 
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_rtf(rtf_path):
    with open(rtf_path, "r", encoding="utf-8") as f:
        return f.read()
    
def generate_jsonl(JSONL_PATH, INPUT_DIR, OUTPUT_DIR):
    entries = []
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith(".png"):
            continue

        base_name = os.path.splitext(file)[0]
        image_path = os.path.join(INPUT_DIR, file)
        rtf_path = os.path.join(OUTPUT_DIR, f"{base_name.replace('Input', 'Output')}.rtf")

        if not os.path.exists(rtf_path):
            print(f"[!] Skipping {file}, no matching RTF.")
            continue

        image_b64 = encode_image_base64(image_path)
        rtf_text = read_rtf(rtf_path)
        formatted_text = rtf_to_text(rtf_text)

        prompt_text = random.choice(USER_PROMPT_VARIANTS) #Get a random prompt for this image. Adds variation to training

        entry = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": formatted_text
                }
            ]
        }

        entries.append(entry)

    with open(JSONL_PATH, "w", encoding="utf-8") as out_file:
        for entry in entries:
            out_file.write(json.dumps(entry) + "\n")

    print(f"[✓] Generated {len(entries)} entries at {JSONL_PATH}")
    return JSONL_PATH



def generateTrainingFile():
    filePath = generate_jsonl(TRAINING_JSONL_PATH, TRAINING_INPUT_DIR, TRAINING_OUTPUT_DIR)
    return filePath

def generateValidationFile():
    filePath = generate_jsonl(VALIDATION_JSONL_PATH, VALIDATION_INPUT_DIR, VALIDATION_OUTPUT_DIR)
    return filePath