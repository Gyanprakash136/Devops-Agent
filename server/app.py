import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from .environment import CloudOptimizerEnv, Action, Observation, Reward
from .ui import create_ui

app = FastAPI(title="OpenEnv Cloud Infrastructure Cost Optimizer")

# Global dict to store envs by task name (for simple statefulness if needed)
envs = {}

def get_env(task_name: str) -> CloudOptimizerEnv:
    if task_name not in envs:
        envs[task_name] = CloudOptimizerEnv(task_name=task_name)
    return envs[task_name]

class StepRequest(BaseModel):
    task: str = "easy"
    action: Action

@app.post("/reset", response_model=Observation)
def reset_env(task: str = "easy"):
    env = get_env(task)
    return env.reset()

@app.post("/step", response_model=Reward)
def step_env(req: StepRequest):
    env = get_env(req.task)
    return env.step(req.action)

@app.get("/state", response_model=Observation)
def state_env(task: str = "easy"):
    env = get_env(task)
    return env.state()

from fastapi.responses import HTMLResponse

@app.get("/")
def home():
    return HTMLResponse("<html><head><script>window.location.href='/dashboard';</script></head><body><p>Redirecting to dashboard...</p></body></html>")

@app.get("/health")
def health():
    return {"status": "healthy"}

# Initialize and mount Gradio UI (Mounting at /dashboard to avoid shadowing /reset)
ui = create_ui(envs)
app = gr.mount_gradio_app(app, ui, path="/dashboard")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

