import openai
import base64
import requests
import io
from dotenv import load_dotenv
import os
import gradio as gr
from pathlib import Path
import json
import feedback


load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY") #set auth key for Open AI

#System prompt that goes to llm
SYSTEM_PROMPT = 'You are trying to generate prompts for these images. Your job is to describe the image and all its functions/designs in a way that LLMs or other models can develop seamless UI components from it.'

#Feedback URL
FEEDBACK_URL = os.getenv("FEEDBACK_URL")

#Most recent output
GENERATED_OUTPUT = ''

# Convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_model_config():
    CONFIG_PATH = Path("model_config.json")

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    else:
        return {"current_model_id": None, "model_history": []}


#SET Most Recent Model
def getMostRecentModel():
    config = load_model_config()
    MODEL_NAME = config['current_model_id']
    print(f"using model {MODEL_NAME} for chat completion.")
    return MODEL_NAME


# Get model response
def generate_prompt(user_text, image):

    global GENERATED_OUTPUT #ref the global output


    if not user_text and not image:
        return "⚠️ Please provide a text prompt, an image, or both."
    
    #Get most recent model
    MODEL = getMostRecentModel()

    messages = [
        {
        "role": "system",
        "content": [
            {
            "text": "You are Bob, a senior prompt engineer. \n\nYou are an assistant trained to generate structured responses from UI screenshots. \n\nYou are trying to generate prompts for these images. Your job is to describe the image and all its functions/designs in a way that LLMs or other models can develop seamless UI components from it.\n\nAlways output clean, parsable markdown.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "text": "You are a UI-to-metadata assistant. For every input, generate clean markdown using the following format:\n\nSections - <Comma-separated list of UI components>\n\n**<ComponentName>**\n-- Detail 1\n-- Detail 2\n\nNotes, at the end\n\nAlso, remember that you are generating prompts for React and Next, most likely. So add prompts to generate default routerlinks or actions that the user can change later if needed, whenever necessary. \n\nYou are also generating this for code generation engines down the line, so describing alignments, fonts, colours, UI elements used, libraries used etc is important. Do so whenever necessary. \n\nAlways return structured markdown output only.\n",
            "type": "text"
            }
        ]
        }
    ]

    user_content = []

    if user_text:
        user_content.append({ "type": "text", "text": user_text })

    if image:
        image_b64 = image_to_base64(image)
        user_content.append({ "type": "image_url", "image_url": { "url": f"data:image/png;base64,{image_b64}" } })

    messages.append({
        "role": "user",
        "content": user_content
    })

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "text"
        },
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    GENERATED_OUTPUT = response.choices[0].message.content.strip()
    return GENERATED_OUTPUT

#Store feedback for retraining
def submit_feedback(text_prompt, image_input, output_box, feedback_box):

    if not image_input or GENERATED_OUTPUT == output_box:
        return "⚠️ Learning updates must include an image and changes to the output."

    image_b64 = image_to_base64(image_input)

    payload = {
        "text_prompt": text_prompt,
        "image_base64": image_b64,
        "generated_output": GENERATED_OUTPUT,
        "corrected_output": output_box,
        "feedback_comment": feedback_box
    }

    try:
        res = requests.post(FEEDBACK_URL, json=payload)
        if res.ok:
            return "✅ Feedback submitted!"
        else:
            return f"Error: {res.status_code} - {res.text}"
    except Exception as e:
        return f"❌ Exception: {str(e)}"




def chatUI():
    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 Bob, the UI-to-Prompt Generator (he's learning)")

        with gr.Row():
            text_prompt = gr.Textbox(label="Describe what you want (Optional)", placeholder="e.g. Hey! Generate this prompt.")
            image_input = gr.Image(type="pil", label="Upload a UI Screenshot (Optional)")

        with gr.Row():
            output_box = gr.Textbox(label="Model Output (Modfiy for correction)", lines=12)
            feedback_box = gr.Textbox(label="Any Feedback (Optional)", lines=12, placeholder="Enter additional comments here if needed.")

        with gr.Row():
            generate_btn = gr.Button("🔍 Generate Prompt")
            submit_btn = gr.Button("✅ Submit Feedback")

        alert_output = gr.HTML()

        generate_btn.click(fn=generate_prompt, inputs=[text_prompt, image_input], outputs=output_box)
        submit_btn.click(fn=submit_feedback, inputs=[text_prompt, image_input, output_box, feedback_box], outputs=alert_output)


    demo.launch()

if __name__ == "__main__":
    chatUI()