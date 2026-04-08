import os
import json
import traceback
from traceback import format_exc
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# The Hackathon specifically asks for HF_TOKEN as the primary API key
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "cloud_optimizer")

MAX_STEPS = 10

# In a real OpenEnv validation setup with docker, we would connect to HTTP,
# but since the sample inference uses direct Python import, we will follow that,
# or we can do REST calls to the FastAPI app. The guidelines say "sample inference script uses direct imports",
# but also mentioned HF Space. Wait. The provided Sample Inference script imports MyEnvV4Env! 
# Let's just import the environment locally for evaluation.
from server.environment import CloudOptimizerEnv, Action

def mock_agent_call(task_name: str, obs: any):
    """Rule-based logic to solve tasks when LLM fails or for local debugging."""
    servers = [s for s in obs.servers if s.status == "running"]
    
    if task_name == "easy":
        # Find all servers with exactly 0% CPU
        zombies = [s for s in servers if s.cpu_utilization == 0.0]
        if zombies:
            return {"server_id": zombies[0].server_id, "action_type": "terminate"}
            
    elif task_name == "medium":
        # Find servers with < 10% CPU and downsize them
        # We need to find one that isn't already the smallest (t3.micro)
        underutilized = [s for s in servers if s.cpu_utilization < 0.10 and s.instance_type != "t3.micro"]
        if underutilized:
            return {"server_id": underutilized[0].server_id, "action_type": "downsize"}

            
    elif task_name == "hard":
        # Strategy: Terminate zombies first, then downsize underutilized, 
        # ensuring we stay above min_compute_capacity.
        total_vcpu = sum(s.vcpu for s in obs.servers if s.status == "running")
        
        # 1. Terminate zeros (safest)
        zombies = [s for s in servers if s.cpu_utilization == 0.0]
        if zombies and total_vcpu > obs.min_compute_capacity:
             return {"server_id": zombies[0].server_id, "action_type": "terminate"}
             
        # 2. Downsize underutilized if budget is still tight
        total_cost = sum(s.cost_per_month for s in obs.servers if s.status == "running")
        if total_cost > obs.budget:
            underutilized = [s for s in servers if s.cpu_utilization < 0.10]
            if underutilized:
                return {"server_id": underutilized[0].server_id, "action_type": "downsize"}

    return {"server_id": servers[0].server_id if servers else "none", "action_type": "none"}

def run_task(task_name: str, client: OpenAI):
    env = CloudOptimizerEnv(task_name=task_name)
    obs = env.reset()
    
    # Initialize variables for cleanup/finally block
    step_count = 0
    done = False
    success = False
    rewards_history = []
    final_score = 0.0
    
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
    
    try:
        for _ in range(MAX_STEPS):
            step_count += 1
            action_str = ""
            error_msg = "null"
            reward_val = 0.0
            
            # Formulate the prompt
            sys_prompt = (
                "You are a DevOps agent managing a cloud cluster. Your goal is to optimize costs based on the task description.\n"
                "Action space: { \"server_id\": \"...\", \"action_type\": \"terminate\" | \"downsize\" | \"none\" }\n"
                "Return MUST be valid JSON matching the action space precisely."
            )
            
            user_prompt = f"Current state: {obs.model_dump_json()}\nWhat is your next action in JSON format?"
            
            try:
                # Call LLM
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                content = response.choices[0].message.content
                action_data = json.loads(content)
            except Exception as e:
                # FALLBACK LOGIC
                error_type = "API_ERROR"
                if "insufficient_quota" in str(e):
                    error_type = "QUOTA_EXCEEDED"
                
                print(f"[INFO] {error_type}: Falling back to Mock Agent local logic.")
                action_data = mock_agent_call(task_name, obs)
                error_msg = f"Fallback({error_type})"

            try:
                action_obj = Action(**action_data)
                action_str = f"action_type='{action_obj.action_type}',server_id='{action_obj.server_id}'"
                
                # Step the environment
                reward_obj = env.step(action_obj)
                obs = env.state()
                
                reward_val = reward_obj.reward
                done = reward_obj.done
                
                if reward_obj.info and "score" in reward_obj.info:
                    final_score = float(reward_obj.info["score"])
                
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                done = True
                action_str = "error_action"
            
            rewards_history.append(reward_val)
            
            # Format booleans for STDOUT
            done_str = "true" if done else "false"
            
            # Print [STEP]
            print(f"[STEP] step={step_count} action={action_str} reward={reward_val:.2f} done={done_str} error={error_msg}")
            
            if done:
                success = True 
                break

    except Exception as e:
        success = False
        print(f"CRITICAL ERROR: {str(e)}")
    finally:
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards_history) if rewards_history else "0.00"
        
        final_score = max(0.0, min(1.0, final_score))
        print(f"[END] success={success_str} steps={step_count} score={final_score:.2f} rewards={rewards_str}")


def main():
    if not API_BASE_URL:
        print("API_BASE_URL must be defined in environment")
        return
        
    client = OpenAI(api_key=API_KEY or "dummy_key", base_url=API_BASE_URL)
    
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_task(task, client)

if __name__ == "__main__":
    main()
