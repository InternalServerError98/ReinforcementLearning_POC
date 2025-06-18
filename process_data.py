import os
import base64
import json
import random
from striprtf.striprtf import rtf_to_text
from dotenv import load_dotenv

load_dotenv()  # ✅ This loads the .env file into the environment

TRAINING_INPUT_DIR = os.getenv("INPUT_FILE_PATH")
TRAINING_OUTPUT_DIR = os.getenv("OUTPUT_FILE_PATH")

VALIDATION_INPUT_DIR = os.getenv("VALIDATION_INPUT_PATH")
VALIDATION_OUTPUT_DIR = os.getenv("VALIDATION_OUTPUT_PATH")

TRAINING_JSONL_PATH = 'training_data.jsonl'
VALIDATION_JSONL_PATH = 'validation_data.jsonl'

# Add variations to the user prompt to make it more diverse, if you are using explicit user prompts later.
USER_PROMPT_VARIANTS = [
    "Generate a prompt for this."
]

#SYSTEM_PROMPT = "You are a lead prompt Engineer. You are trying to generate prompts from the UI images presented to you. Your job is to describe the image and all its funcions/designs in a way that LLMs or other models can develop seamless UI components from it."

MessageHead = [
    {
      "role": "system",
      "content": [
        {
          "text": "You are a senior prompt engineer. You convert designs to prompts that help LLMs translate them into code.",
          "type": "text"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "text": "Remember to adhere to the following metrics : \n\nYou have to recursively break down the page into smaller elements while describing their position and functionality. \n\nFollow this naming convention\n\nSection - [Section 1, Section 2]\n\n**Section 1**\n-- Generic Desc\n****SubSection 1.1****\n****SubSection 1.2****\n\n**Section 2**\n.......\n\nAnd so on. ",
          "type": "text"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "Okay !",
          "type": "text"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "text": "Always respond in markdown, always!",
          "type": "text"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "Understood. ",
          "type": "text"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "text": "In terms of description, apart from how the page looks you must describe functionality. \n\nButtons: must add prompts to generate place holder on clicks. \nLinks: must have placeholder routers.\nSelectboxes: Must have placeholder loaders. \n\nYou must add these for other associated components.\n",
          "type": "text"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "Understood. ",
          "type": "text"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "text": "Lastly, think like a developer. Implement lazy loading, scrolling to bottom, responsiveness etc whenever required. ",
          "type": "text"
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "text": "Got it, I am ready now.",
          "type": "text"
        }
      ]
    }
  ]


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

        MessageHead.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                            ]
                        } )
        
        MessageHead.append(
                         {
                            "role": "assistant",
                            "content": formatted_text
                         } 
                        )
        entry = {


            "messages": MessageHead
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