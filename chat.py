import base64
import requests
import io
import gradio as gr
from PIL import Image
import Env.configuration as configuration
import Services.chat_completion as chat_completion

#Most recent output
GENERATED_OUTPUT = ''

# Convert image to base64
def image_to_base64(image):
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL Image.")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Get model response
def generate_prompt(user_text, image):
    

    global GENERATED_OUTPUT #ref the global output
    if not image:
        return "⚠️ Please provide an image."
    

    #append message header to message body
    messages = configuration.secrets.MESSAGE_HEADER

    #add the user prompt and image to the message
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

    #generate a response based on whether we are using OpenAI or Azure OpenAI
    response = chat_completion.GenerateResponse(messages) 
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
        res = requests.post(configuration.secrets.FEEDBACK_URL, json=payload)
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


