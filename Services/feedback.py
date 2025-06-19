from fastapi import FastAPI
from pydantic import BaseModel
import json
import uvicorn
import Env.configuration as configuration


MESSAGE_HEAD = [
        {
        "role": "system",
        "content": [
            {
            "text": "You are Bob, a senior prompt engineer. \n\nYou are an assistant trained to generate structured prompts for LLM's from UI screenshots. \n\nAlways output clean, parsable markdown.",
            "type": "text"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "text": "You are a UI-to-prompt assistant. For every input, generate clean markdown using the following format:\n\nSections - <Comma-separated list of UI components>\n\n**<ComponentName>**\n-- Detail 1\n-- Detail 2\n\nNotes, at the end\n\nAlso, remember that you are generating prompts for React and Next, most likely. So add prompts to generate default routerlinks or actions that the user can change later if needed, whenever necessary. \n\nYou are also generating this for code generation engines down the line, so describing alignments, fonts, colours, UI elements used, libraries used etc is important. Do so whenever necessary. \n\nAlways return structured markdown output only.\n",
            "type": "text"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "text": "Understood, give me an image and I will generate a prompt in no time !",
            "type": "text"
            }
        ]
        }
]


app = FastAPI()
FEEDBACK_FILE = configuration.secrets.FEEDBACK_FILE

class Feedback(BaseModel):
    text_prompt: str
    image_base64: str
    generated_output: str
    corrected_output: str
    feedback_comment: str = ""  # Optional


@app.post("/feedback")
async def collect_feedback(feedback: Feedback):

    record = MESSAGE_HEAD


    #Initial User Prompt
    user_content = []
    user_content.append({ "type": "text", "text": feedback.text_prompt })
    user_content.append({ "type": "image_url", "image_url": { "url": f"data:image/png;base64,{feedback.image_base64}" } })

    record.append({
        "role": "user",
        "content": user_content
    })

    #Initial Response
    record.append({

        "role": "assistant",
        "content": [
            {
            "text": feedback.generated_output,
            "type": "text"
            }
        ]
    })

    #Feedback given by user
    record.append({

        "role": "user",
        "content": [
            {
            "text": "This is not accurate, primarily because of this. " + feedback.feedback_comment,
            "type": "text"
            }
        ]
    })

    #Train behavior to generate learning
    record.append({

        "role": "assistant",
        "content": [
            {
            "text": "Understood, here is the correct feedback. \n" + feedback.corrected_output,
            "type": "text"
            }
        ]
    })

    #Closing user comment
    record.append({

        "role": "user",
        "content": [
            {
            "text": "Perfect, remember to learn and incorporate this into your next response as well.",
            "type": "text"
            }
        ]
    })

    #Assistant Understands
     #Closing user comment
    record.append({

        "role": "assistant",
        "content": [
            {
            "text": "Abosultely, give me another image and I will be ready.",
            "type": "text"
            }
        ]
    })

    feedback_entry = {"messages" : record}

    #Enter feedback loop into json file
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")

    return {"status": "ok", "message": "Feedback stored."}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)