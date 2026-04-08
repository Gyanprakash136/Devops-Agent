---
title: Cloud Infrastructure Cost Optimizer
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# OpenEnv: Cloud Infrastructure Cost Optimizer

An environment designed for the OpenEnv Hackathon where an AI agent acts as a DevOps engineer to optimize a fleet of cloud servers. The environment returns real-world metrics like CPU utilization, instance types, and cost, challenging the agent to minimize cost while maintaining sufficient compute capacity.

## Environment Description

This environment simulates a real-world DevOps task: cloud resource optimization. In modern infrastructure, "zombie" servers (0% CPU) and "underutilized" servers (<10% CPU) waste thousands of dollars. An agent must identify these servers and either terminate or downsize them to reach a budget goal without compromising the total compute capacity (vCPUs) required for the system to function.

## Observation Space

The `Observation` Pydantic model provides a list of `ServerState` objects.
```json
{
  "servers": [
    {
      "server_id": "i-01234abcd",
      "instance_type": "t3.medium",
      "cpu_utilization": 0.5,
      "cost_per_month": 30.0,
      "status": "running",
      "vcpu": 2
    }
  ],
  "budget": 500.0,
  "min_compute_capacity": 10
}
```

## Action Space

The `Action` Pydantic model requires the agent to specify the server and the action.
```json
{
  "server_id": "i-01234abcd",
  "action_type": "terminate"
}
```
*Options for `action_type`: "terminate", "downsize", "none"*

## Tasks & Grading Logic

### Easy: Terminate Zombie Servers
- **Objective:** Find and terminate all servers with exactly 0% CPU utilization.
- **Grader:** +1.0 for correctly terminating all zombies and none of the active servers.

### Medium: Downsize Underutilized Servers
- **Objective:** Find instances running at <10% capacity and downsize them to a smaller instance tier.
- **Grader:** +1.0 for correctly downsizing all <10% servers. Penalized if servers with >=10% CPU are modified.

### Hard: Complete Cluster Optimization
- **Objective:** Optimize a mixed cluster to get total monthly cost under $500 while maintaining a minimum total compute capacity (10 vCPUs).
- **Grader:** Score from 0.0 to 1.0 based on how well the budget is met without dropping below standard compute minimums.

## Baseline Scores

| Task | Score |
| :--- | :--- |
| Easy | 1.0 |
| Medium | 1.0 |
| Hard | 1.0 |

## Setup & Running Locally

1. **Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Environment & Dashboard**:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 7860
   ```

3. **Run Baseline Inference**:
   ```bash
   export HF_TOKEN="your_api_key_here"
   python inference.py
   ```

## Deployment
This project is ready for deployment as a Docker Space on Hugging Face.


