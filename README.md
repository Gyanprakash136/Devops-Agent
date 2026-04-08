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
    <img src="https://img.shields.io/badge/Phase--2-Validated-blue.svg" alt="Phase 2 Validated">
    <img src="https://img.shields.io/badge/Framework-FastAPI-009688.svg" alt="FastAPI">
    <img src="https://img.shields.io/badge/Frontend-Gradio-ff5200.svg" alt="Gradio">
  </p>
</div>

---

## 📖 Overview

The **Cloud Infrastructure Cost Optimizer** is a production-grade simulation environment built for the Meta x Scaler Hackathon. It puts an AI agent in the driver's seat of a cloud fleet, challenging it to optimize monthly spending while maintaining mission-critical compute capacity.

This project goes beyond a simple script; it is a full **Multi-Mode Deployment** compliant repository, featuring a real-time visualization dashboard and robust programmatic grading.

## 🛠️ Core Mechanics

### 1. High-Fidelity Observation Space
The agent perceives the true state of the infrastructure:
- **Fleet Analytics**: Instance ID, Tier (t3/m5/c5), CPU Utilization, and monthly Cost.
- **Constraints**: A global **Budget** and a **Minimum Capacity (vCPU)** floor that must never be breached.

### 2. Strategic Action Space
- `terminate`: Eradicate "Zombie" servers (0% CPU).
- `downsize`: "Right-size" instances to the next smaller tier to eliminate waste.
- `none`: Maintain current configuration.

---

## 🎯 Task Tiers & Grading

| Difficulty | Task Name | Key Performance Indicator (KPI) | Score Range |
| :--- | :--- | :--- | :--- |
| **Easy** | Zombie Hunt | Termination of all 0% CPU servers. | `0.01 - 0.99` |
| **Medium** | Right-Sizing | Mitigation of waste in servers with < 10% CPU. | `0.01 - 0.99` |
| **Hard** | Fleet Optimization | Meet target budget while maintaining capacity. | `0.01 - 0.99` |

> [!NOTE]
> **Safety Clamping**: Following Phase 2 validation requirements, all scores are strictly clamped between `0.01` and `0.99` to ensure smooth agentic evaluation and transparent scoring.

---

## 🖥️ Live Visualization Dashboard

We have integrated a **Gradio-powered Control Center** directly into the environment. 
- **Live Metrics**: Real-time cost and capacity counters.
- **Visual Fleet Status**: Color-coded server status and CPU utilization bars.
- **Direct App URL**: [https://huggingface.co/spaces/gyan0009/devops-agent-cost-optimizer](https://huggingface.co/spaces/gyan0009/devops-agent-cost-optimizer)

---

## 🚀 Usage Guide

### 1. Local Development
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run API & Dashboard
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference Benchmark
```bash
export HF_TOKEN="your_token"
python inference.py
```

---

## ⚖️ Compliance & Reproducibility
- **OpenEnv Spec**: Full compliance with Pydantic models and `openenv.yaml`.
- **Reproducibility**: Includes a **Mock Agent Fallback** to ensure consistent scores regardless of LLM quota limit.
- **Containerization**: Optimized `Dockerfile` provided for seamless scaling.

<div align="center">
  <sub>Managed by Gyan Prakash for the Meta PyTorch Hackathon.</sub>
</div>
