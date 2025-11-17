from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

# Load model on startup
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.1, max_tokens=512)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    output: str

@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    outputs = llm.generate([req.prompt], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text
    return ChatResponse(output=text)