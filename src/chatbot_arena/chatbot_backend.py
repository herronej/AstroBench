import os
import uuid
import uvicorn
import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Cookie, Response
from fastapi.requests import Request
from pydantic import BaseModel
from langchain_community.llms.sambanova import SambaStudio
from dataset_sampler import QuestionSampler 
from vote_logger import init_db, log_vote
from model_sampler import get_next_model_pair

user_sessions = {}  # session_id -> QuestionSampler

app = FastAPI()

@app.middleware("http")
async def add_session_id(request: Request, call_next):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response = await call_next(request)
        response.set_cookie(key="session_id", value=session_id)
        return response
    else:
        return await call_next(request)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Chatbot model titles
chatbot_titles = {
    "model_a": "Model A",
    "model_b": "Model B"
}

# Define input schema
class PromptInput(BaseModel):
    prompt: str

class VoteRequest(BaseModel):
    vote: str

@app.get("/chatbot", response_class=HTMLResponse)
async def render_chatbot_ui(request: Request):
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "model_a": chatbot_titles["model_a"],
        "model_b": chatbot_titles["model_b"],
        #"prompts": json.dumps(prompts)
    })

'''
@app.get("/generate-prompt")
async def generate_prompt(session_id: str = Cookie(None)):
    if session_id not in user_sessions: 
        user_sessions[session_id] = QuestionSampler()
    sampler = user_sessions[session_id]
    next_q = sampler.get_next_question()

    # Save question so we can fetch correct/answer/explanation later
    user_sessions[session_id].last_question = next_q

    print(f"[session {session_id[:8]}...] {next_q['type'].upper()} → {next_q['question'][:80]}")

    return next_q  # return entire question dict
'''

@app.get("/generate-prompt")
async def generate_prompt(session_id: str = Cookie(None)):
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "sampler": QuestionSampler(),
            "model_pair": get_next_model_pair(),
            "count": 0
        }
    session = user_sessions[session_id]
    session["count"] += 1

    # Resample model pair every 2 questions
    if session["count"] % 2 == 1:
        session["model_pair"] = get_next_model_pair()

    next_q = session["sampler"].get_next_question()
    session["last_question"] = next_q

    print(f"[session {session_id[:8]}...] {next_q['type'].upper()} → {next_q['question'][:80]}")
    return next_q


# Store the latest prompt here
latest_prompt = None

@app.post("/save-prompt")
async def save_prompt(data: PromptInput, session_id: str = Cookie(None)):
    global latest_prompt

    # set environment variables with endpoint url and api key
    os.environ["SAMBASTUDIO_URL"] = "https://sambanova.cades.ornl.gov/api/v2/predict/generic/37174856-d664-49f5-bede-eeb956c881b4/8b3a20e7-c841-444a-9000-3aa151d5edfe"
    os.environ["SAMBASTUDIO_API_KEY"] = "b647941d-9160-4095-b639-ecd6b8443cbb"

    latest_prompt = data.prompt
    session = user_sessions.get(session_id)
    model_a_name, model_b_name = session["model_pair"]

    mod_a = SambaStudio(
        model_kwargs={
            "model": model_a_name, 
            "max_tokens": 512, 
            "temperature": 0.3, 
            "top_p": 0.8,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "process_prompt": False
            # "repetition_penalty":  1.0,
            # "top_k": 50,
            # "top_logprobs": 0,
        },
    )

    mod_b = SambaStudio(
        model_kwargs={
            "model": model_b_name, 
            "max_tokens": 512, 
            "temperature": 0.3, 
            "top_p": 0.8,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "process_prompt": False
        },
    )

    model_a_response = mod_a.invoke(latest_prompt)
    model_b_response = mod_b.invoke(latest_prompt)
    print(mod_a.model_kwargs.get("model", "unknown"), ":\n\n")
    print(model_a_response, "\n\n")
    print(mod_b.model_kwargs.get("model", "unknown"), ":\n\n")
    print(model_b_response, "\n\n")
    
    # Save all relevant info so we can use it during /vote
    last_q = session["last_question"]
    session["last_question"] = {
        **last_q,
        "model_a": model_a_name,
        "model_b": model_b_name,
        "model_a_response": model_a_response,
        "model_b_response": model_b_response,
    }

    return {
        "model_a_response": model_a_response,
        "model_b_response": model_b_response,
        "correct": last_q.get("correct", None),
        "answer": last_q.get("answer", None),
        "explanation": last_q.get("explanation", None)
    }


# New endpoint to register vote
@app.post("/vote")
async def receive_vote(vote_request: VoteRequest, session_id: str = Cookie(None)):
    last_q = user_sessions.get(session_id, {}).get("last_question", {})
    
    log_vote(
        session_id=session_id or "anonymous",
        question_type=last_q.get("type", ""),
        question=last_q.get("question", ""),
        correct_answer=last_q.get("correct", last_q.get("answer", "")),
        explanation=last_q.get("explanation", ""),
        model_a=last_q.get("model_a", ""),
        model_a_response=last_q.get("model_a_response", ""),
        model_b=last_q.get("model_b", ""),
        model_b_response=last_q.get("model_b_response", ""),
        vote=vote_request.vote
    )
    print(f"Vote received: {vote_request.vote}", flush=True)  # Later save to database or log
    return {"message": "Vote registered successfully"}

if __name__ == "__main__":
    init_db()
    print("Visit chatbot UI at http://localhost:8001/chatbot")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)