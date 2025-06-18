import openai
import base64
import requests
import io
import os
import gradio as gr
from pathlib import Path
import json
from dotenv import load_dotenv
from PIL import Image



load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY") #set auth key for Open AI
openai.api_type = os.getenv("TRAINING_ENVIRONMENT") #set environment for Open AI

#System prompt that goes to llm
SYSTEM_PROMPT = 'You are a senior prompt engineer. You convert designs to prompts that help LLMs translate them into code.'

#Feedback URL
FEEDBACK_URL = os.getenv("FEEDBACK_URL")

#Most recent output
GENERATED_OUTPUT = ''

# Convert image to base64
def image_to_base64(image):
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL Image.")
    buffered = io.BytesIO()
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


    if not image:
        return "⚠️ Please provide an image."
    
    #Get most recent model
    MODEL = getMostRecentModel()


    messages = [


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
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Generate a prompt for this"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{image_to_base64(Image.open('Data/Train/Input/Input_3.png'))}"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "```markdown\nSection - [Section 1, Section 2]\n\n**Section 1**\n- Generic Desc: Centered login form on a light background with a wave-patterned blue overlay at the bottom. The form is placed over a light grey container with a shadow effect. The interface is clean, flat, and uses a red accent color for actions and highlights. The design should be responsive and scale for both desktop and mobile screens.\n\n****SubSection 1.1****\n- Header area, positioned at the top center of the form container.\n    - Functionality: Static text (\"Login Form\"), styled in bold red typography.\n    - Should scale down on smaller screens for responsiveness.\n\n****SubSection 1.2****\n- Form area, centrally aligned within a raised card/container.\n    - Contains two fields:\n        1. Username Input\n            - Label: \"Username *\"\n            - Placeholder: \"Enter your Username\"\n            - Functionality: Standard text input, required & should accept only text (add validation logic).\n        2. Password Input\n            - Label: \"Password *\"\n            - Placeholder: \"Enter your Password\"\n            - Functionality: Standard password input, required (add validation logic).\n    - Both fields are styled as pill-shaped, with padding.\n\n****SubSection 1.3****\n- Checkbox area, below password input field.\n    - \"Remember me\" option.\n    - Functionality: Simple checkbox, value persists login state (add placeholder logic).\n\n****SubSection 1.4****\n- Action button area, full-width, with rounded corners.\n    - Button Text: \"LOGIN\"\n    - Functionality: OnClick triggers a placeholder function for login (insert login loader/spinner or placeholder function).\n    - Button is disabled until both username and password are filled.\n    - Button uses high-contrast color (red) for visibility.\n\n**Section 2**\n- Footer navigation, aligned horizontally below the button within the card boundary, separated by a contrasting background.\n    - Contains two link elements:\n        1. Don't have an account?\n            - Functionality: Placeholder router to signup/registration page.\n            - Color: Red, with hover effect.\n        2. Forgot password?\n            - Functionality: Placeholder router to forgot/reset password page.\n            - Color: Red, with hover effect.\n    - Horizontal divider (red bar) below both links for separation and emphasis.\n\n**Additional Developer Considerations**\n- Responsive: Form should be vertically and horizontally centered, auto scale and stack on mobile, min-width maintained for readability.\n- Accessibility: Ensure all labels are associated with their inputs, add aria attributes as required.\n- Lazy loading: Component is simple, but in anticipated expansion modularize for future lazy import/SSR.\n- Scrolling: Auto-scroll to card if loaded deep in page.\n- All buttons & links have placeholder loaders or router handlers attached for stub logic.\n```\n"
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
        gr.Markdown("## UI-to-Prompt Generator")

        with gr.Row():
          # LEFT COLUMN: Inputs
          with gr.Column(scale=1):
            text_prompt = gr.Textbox(
               label="Default Value",
               value="Generate a prompt for this.",
               visible=False,  # Hidden by default
              ) 
            image_input = gr.Image(type="pil", label="🖼️ Upload UI Screenshot (Optional)")
            feedback_box = gr.Textbox(label="📝 Feedback (Optional)", lines=6, placeholder="Enter any feedback or corrections here...")

          # RIGHT COLUMN: Model Output
          with gr.Column(scale=1):
            output_box = gr.Textbox(label="💡 Model Output (You can edit this)", lines=25)

        with gr.Row():
            generate_btn = gr.Button("🔍 Generate Prompt")
            submit_btn = gr.Button("✅ Submit Feedback")

        alert_output = gr.HTML()

        generate_btn.click(fn=generate_prompt, inputs=[text_prompt,image_input], outputs=output_box)
        submit_btn.click(fn=submit_feedback, inputs=[text_prompt,image_input, output_box, feedback_box], outputs=alert_output)


    demo.launch()

if __name__ == "__main__":
    chatUI()


