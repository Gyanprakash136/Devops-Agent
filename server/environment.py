import random
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# --- Pydantic Models ---

class ServerState(BaseModel):
    server_id: str
    instance_type: str
    cpu_utilization: float = Field(..., description="Percentage in [0, 1]")
    cost_per_month: float
    status: Literal["running", "terminated"]
    vcpu: int

class Observation(BaseModel):
    servers: List[ServerState]
    budget: float
    min_compute_capacity: int

class Action(BaseModel):
    server_id: str
    action_type: Literal["terminate", "downsize", "none"]

class Reward(BaseModel):
    reward: float
    done: bool
    info: Optional[dict] = None

# --- Constants ---
INSTANCE_SPECS = {
    "t3.micro": {"cost": 7.5, "vcpu": 2, "next_down": None},
    "t3.small": {"cost": 15.0, "vcpu": 2, "next_down": "t3.micro"},
    "t3.medium": {"cost": 30.0, "vcpu": 2, "next_down": "t3.small"},
    "m5.large": {"cost": 70.0, "vcpu": 2, "next_down": "t3.medium"},
    "m5.xlarge": {"cost": 140.0, "vcpu": 4, "next_down": "m5.large"},
    "c5.2xlarge": {"cost": 245.0, "vcpu": 8, "next_down": "m5.xlarge"}
}

# --- Environment Logic ---

class CloudOptimizerEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.servers = []
        self.max_steps = 10
        self.current_step = 0
        self.budget = 500.0
        self.min_capacity = 10
        self.initial_state = []
        self.reset()
        
    def _generate_servers(self):
        self.servers = []
        # Deterministic generation for reliable grading
        random.seed(42) 
        
        if self.task_name == "easy":
            # 2 zombies, 3 normal
            types = ["t3.medium", "m5.large", "c5.2xlarge", "t3.small", "m5.xlarge"]
            cpus = [0.0, 0.45, 0.0, 0.8, 0.6]
        elif self.task_name == "medium":
            # 3 underutilized (< 10%), 2 normal
            types = ["m5.xlarge", "c5.2xlarge", "m5.large", "t3.medium", "t3.micro"]
            cpus = [0.05, 0.08, 0.7, 0.02, 0.9]
        else: # hard
            # Mixed cluster, initial cost high
            types = ["c5.2xlarge", "m5.xlarge", "m5.xlarge", "m5.large", "t3.medium", "t3.medium", "t3.small"]
            cpus = [0.3, 0.05, 0.7, 0.0, 0.0, 0.8, 0.9]

        for i, (t, c) in enumerate(zip(types, cpus)):
            self.servers.append(ServerState(
                server_id=f"i-{1000+i}",
                instance_type=t,
                cpu_utilization=c,
                cost_per_month=INSTANCE_SPECS[t]["cost"],
                status="running",
                vcpu=INSTANCE_SPECS[t]["vcpu"]
            ))
        
        # Deep copy to track initial state for grading
        self.initial_state = [s.model_copy() for s in self.servers]

    def reset(self) -> Observation:
        self._generate_servers()
        self.current_step = 0
        return self.state()

    def state(self) -> Observation:
        return Observation(
            servers=self.servers,
            budget=self.budget,
            min_compute_capacity=self.min_capacity
        )

    def step(self, action: Action) -> Reward:
        self.current_step += 1
        
        # Apply action
        server = next((s for s in self.servers if s.server_id == action.server_id), None)
        
        if server is None:
            # Invalid server
            return self._calculate_reward_and_done(invalid_action=True)

        if server.status == "terminated" and action.action_type != "none":
            # Action on terminated server
            return self._calculate_reward_and_done(invalid_action=True)

        if action.action_type == "terminate":
            server.status = "terminated"
            server.cost_per_month = 0.0
            server.vcpu = 0
        elif action.action_type == "downsize":
            next_type = INSTANCE_SPECS[server.instance_type]["next_down"]
            if next_type:
                server.instance_type = next_type
                server.cost_per_month = INSTANCE_SPECS[next_type]["cost"]
                server.vcpu = INSTANCE_SPECS[next_type]["vcpu"]
            else:
                # Cannot downsize further
                return self._calculate_reward_and_done(invalid_action=True)

        # Evaluate step result
        return self._calculate_reward_and_done()

    def _calculate_reward_and_done(self, invalid_action: bool = False) -> Reward:
        if invalid_action:
            # Immediate fail and episode over on invalid/destructive actions
            # Or just wait for max steps. But usually, invalid action drops score or wastes a step.
            pass # Continue and check at the end. Or we can terminate early.
            
        done = self.current_step >= self.max_steps
        
        score = 0.0
        
        if self.task_name == "easy":
            score = self._grade_easy()
            # If everything is resolved correctly, we can finish early
            zombies = [s for s in self.initial_state if s.cpu_utilization == 0.0]
            current_zombies = [s for s in self.servers if s.server_id in [z.server_id for z in zombies]]
            if all(z.status == "terminated" for z in current_zombies):
                done = True
                
        elif self.task_name == "medium":
            score = self._grade_medium()
            underauth = [s for s in self.initial_state if s.cpu_utilization > 0.0 and s.cpu_utilization < 0.10]
            current_underauth = [s for s in self.servers if s.server_id in [z.server_id for z in underauth]]
            # Finish if all underauth have changed instance type
            if all(c.instance_type != u.instance_type for c, u in zip(current_underauth, underauth)):
                 done = True
        else: # hard
            score = self._grade_hard()
        
        # Intermediate rewards are proportional to final score, roughly.
        # But for score in [0, 1], we just output the current evaluated score as reward, 
        # or differences in score.
        # Let's just output the current total score if done, 0 otherwise, or partial score.
        reward = score if done else 0.0

        return Reward(reward=reward, done=done, info={"score": score})

    def _grade_easy(self) -> float:
        # Grade in [0, 1]
        zombies_id = [s.server_id for s in self.initial_state if s.cpu_utilization == 0.0]
        normal_id = [s.server_id for s in self.initial_state if s.cpu_utilization > 0.0]
        
        zombies = [s for s in self.servers if s.server_id in zombies_id]
        normals = [s for s in self.servers if s.server_id in normal_id]
        
        # Penalize killing normal servers heavily
        if any(n.status == "terminated" for n in normals):
            return 0.01
            
        terminated_zombies = sum(1 for z in zombies if z.status == "terminated")
        score = float(terminated_zombies) / max(1, len(zombies_id))
        return min(max(score, 0.01), 0.99)

    def _grade_medium(self) -> float:
        # Grade in [0, 1]
        under_id = [s.server_id for s in self.initial_state if s.cpu_utilization > 0.0 and s.cpu_utilization < 0.10]
        normal_id = [s.server_id for s in self.initial_state if s.server_id not in under_id]
        
        under = [s for s in self.servers if s.server_id in under_id]
        normals = [s for s in self.servers if s.server_id in normal_id]
        
        # Penalize touching normal servers
        if any(n.status == "terminated" or n.instance_type != next(i.instance_type for i in self.initial_state if i.server_id == n.server_id) for n in normals):
            return 0.01
            
        # Penalize terminating underutilized instead of downsizing
        if any(u.status == "terminated" for u in under):
            return 0.01
            
        downsized = sum(1 for u in under if u.instance_type != next(i.instance_type for i in self.initial_state if i.server_id == u.server_id))
        score = float(downsized) / max(1, len(under_id))
        return min(max(score, 0.01), 0.99)

    def _grade_hard(self) -> float:
        # Grade in [0.0, 1.0]
        # Optimizing cost under budget while vcpu >= min_capacity
        # Penalize terminating servers that have > 10% cpu
        high_cpu_ids = [s.server_id for s in self.initial_state if s.cpu_utilization >= 0.10]
        high_cpu = [s for s in self.servers if s.server_id in high_cpu_ids]
        
        if any(h.status == "terminated" for h in high_cpu):
            return 0.01
            
        total_cost = sum(s.cost_per_month for s in self.servers)
        total_vcpu = sum(s.vcpu for s in self.servers)
        
        if total_vcpu < self.min_capacity:
            return 0.01
            
        if total_cost <= self.budget:
            # Met budget and capacity, maximum score!
            return 0.99
            
        initial_cost = sum(s.cost_per_month for s in self.initial_state)
        # Cost reduced but not below budget
        savings = max(0, initial_cost - total_cost)
        max_possible_savings = initial_cost - self.budget
        if max_possible_savings <= 0:
            return 0.99
            
        score = float(savings) / max_possible_savings
        return min(max(score, 0.01), 0.99)
