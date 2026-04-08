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

<div align="center">
  <h1>☁️ Cloud Infrastructure Cost Optimizer</h1>
  <p><b>An OpenEnv Reinforcement Learning Environment for DevOps Automation</b></p>
  
  <p>
    <img src="https://img.shields.io/badge/OpenEnv-Compatible-green.svg" alt="OpenEnv Compatible">
    <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/Frontend-Gradio-ff5200.svg" alt="Gradio">
    <img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License">
  </p>
</div>

---

## 📖 Overview

The **Cloud Infrastructure Cost Optimizer** is a real-world simulation environment designed for the OpenEnv Hackathon. It challenges AI agents to take on the role of a **SRE/DevOps Engineer** managing a large-scale cloud cluster. 

In modern infrastructure, "Zombie" servers and over-provisioned instances cost companies billions. This environment provides a platform to train agents that can autonomously optimize infrastructure costs while ensuring zero downtime and maintaining strict compute capacity requirements.

## 🛠️ Core Mechanics

The environment simulates a fleet of servers with varying CPU utilization, instance types, and monthly costs. The agent must make sequential decisions to move the cluster toward a budget goal.

### Observation Space
The agent receives a full snapshot of the cluster state:
- **Server list**: ID, Instance Type (e.g., `m5.large`), CPU Utilization, and Operational Status.
- **Constraints**: Monthly Budget ($) and Minimum Compute Capacity (vCPUs).

### Action Space
- `terminate`: Shut down an instance (Cost → $0, vCPU → 0).
- `downsize`: Move an instance to the next smaller tier (e.g., `m5.xlarge` → `m5.large`).
- `none`: Maintain current state.

---

## 🎯 Task Definitions & Reward Shaping

The environment includes three tiers of difficulty with programmatic graders that evaluate performance from `0.0` to `1.0`.

| Task | Objective | Target | Reward Logic |
| :--- | :--- | :--- | :--- |
| **Easy** | Zombie Hunt | Terminate 0% CPU servers | Binary success for each zombie terminated. |
| **Medium** | Right-Sizing | Downsize underutilized servers | Proportional reward based on reduction in waste. |
| **Hard** | Cluster-Wide Optimization | Meet budget under constraints | Final score based on budget margin vs. capacity safety. |

**Reward Signal**: The environment provides a dense reward signal that penalizes "destructive" actions (terminating active servers) and rewards incremental cost savings.

---

## 📊 Visual Dashboard

This environment comes integrated with a **Real-Time Gradio Dashboard** (served at `/`). You can watch the agent's progress, trigger manual resets, and inspect individual server metrics through a high-fidelity web interface.

---

## 🚀 Getting Started

### Local Setup
Ensure you have a virtual environment activated:
```bash
# 1. Activate venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Running the Environment
The environment and dashboard run on a single FastAPI port:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running Baseline Inference
The `inference.py` script includes a **Smart Fallback** mode that ensures the script runs successfully even if your OpenAI quota is exceeded.
```bash
export HF_TOKEN="your_token_here"
python inference.py
```

---

## 🐳 Docker Deployment

To build and run the containerized environment:
```bash
docker build -t cloud-optimizer .
docker run -p 7860:7860 cloud-optimizer
```

---

<div align="center">
  <sub>Built for the OpenEnv Round 1 Hackathon.</sub>
</div>
