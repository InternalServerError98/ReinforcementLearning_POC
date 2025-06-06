from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import uvicorn




app = FastAPI()
load_dotenv()
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE")

class Feedback(BaseModel):
    image_base64: str
    generated_output: str
    corrected_output: str
    feedback_comment: str = ""  # Optional


@app.post("/feedback")
async def collect_feedback(feedback: Feedback):
    record = feedback.dict()
    record["timestamp"] = datetime.utcnow().isoformat()

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"status": "ok", "message": "Feedback stored."}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)