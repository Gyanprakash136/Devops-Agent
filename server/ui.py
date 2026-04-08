import gradio as gr
import pandas as pd
from .environment import CloudOptimizerEnv

def create_ui(envs: dict):
    """
    Creates the Gradio UI and connects it to the existing environment instances.
    """
    
    def get_status_data(task_name):
        env = envs.get(task_name)
        if not env:
            return "No data", [], 0, 0
            
        obs = env.state()
        history = [s.model_dump() for s in obs.servers]
        df = pd.DataFrame(history)
        
        # Calculate totals
        total_cost = sum(s.cost_per_month for s in obs.servers)
        total_vcpu = sum(s.vcpu for s in obs.servers)
        
        status_text = f"### Status for {task_name.capitalize()} Task\n"
        status_text += f"**Current Step:** {env.current_step} / {env.max_steps}\n"
        status_text += f"**Budget:** ${total_cost:.2f} / ${obs.budget:.2f}\n"
        status_text += f"**Compute:** {total_vcpu} / {obs.min_compute_capacity} vCPUs\n"
        
        if total_cost > obs.budget:
            status_text += "⚠️ **Over Budget!**"
        if total_vcpu < obs.min_compute_capacity:
            status_text += "❌ **Critical: Not enough capacity!**"
            
        return status_text, df, total_cost, total_vcpu

    def reset_task(task_name):
        env = envs.get(task_name, CloudOptimizerEnv(task_name=task_name))
        envs[task_name] = env
        env.reset()
        return get_status_data(task_name)

    with gr.Blocks(title="Cloud Optimizer Dashboard") as demo:
        gr.Markdown("# ☁️ Cloud Infrastructure Cost Optimizer")
        gr.Markdown("Visualize your cloud cluster and watch the AI agent optimize costs in real-time.")
        
        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"], 
                value="easy", 
                label="Select Task"
            )
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=1):
                status_md = gr.Markdown("Select a task to see metrics.")
            with gr.Column(scale=2):
                cost_gauge = gr.Number(label="Total Monthly Cost ($)")
                vcpu_gauge = gr.Number(label="Total Capacity (vCPUs)")
        
        server_table = gr.Dataframe(
            label="Server Fleet Status",
            headers=["ID", "Type", "CPU %", "Cost/mo", "Status", "vCPUs"],
            interactive=False
        )
        
        # Update function
        def refresh(task):
            return get_status_data(task)

        # Event handlers
        task_dropdown.change(refresh, inputs=[task_dropdown], outputs=[status_md, server_table, cost_gauge, vcpu_gauge])
        reset_btn.click(reset_task, inputs=[task_dropdown], outputs=[status_md, server_table, cost_gauge, vcpu_gauge])
        
        # Periodic refresh to catch agent steps
        demo.load(refresh, inputs=[task_dropdown], outputs=[status_md, server_table, cost_gauge, vcpu_gauge])

    return demo
