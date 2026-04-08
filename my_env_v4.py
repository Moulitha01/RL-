from pydantic import BaseModel 
from typing import Dict


class MyEnvV4Action(BaseModel):
    decision: str   # approve | reject | flag
    reasoning: str


class Observation(BaseModel):
    request_id: str
    employee_role: str
    item: str
    cost: float
    policy: str


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool


class MyEnvV4Env:

    def __init__(self):
        self.step_count = 0

        self.tasks = [
            # EASY
            {
                "id": "PR-1",
                "role": "Intern",
                "item": "AI Subscription",
                "cost": 150,
                "policy": "Interns cannot spend more than 100",
                "correct": "reject"
            },
            # MEDIUM
            {
                "id": "PR-2",
                "role": "Manager",
                "item": "Office Supplies",
                "cost": 800,
                "policy": "Managers can spend up to 1000",
                "correct": "approve"
            },
            # HARD
            {
                "id": "PR-3",
                "role": "Junior Dev",
                "item": "Gaming Laptop",
                "cost": 2000,
                "policy": "Junior Dev limit 500, gaming items restricted",
                "correct": "flag"
            }
        ]

        self.current_task = 0

    @classmethod
    async def from_docker_image(cls, image_name):
        return cls()

    async def reset(self):
        self.step_count = 0
        task = self.tasks[self.current_task]

        return StepResult(
            observation=Observation(
                request_id=task["id"],
                employee_role=task["role"],
                item=task["item"],
                cost=task["cost"],
                policy=task["policy"]
            ),
            reward=0.0,
            done=False
        )

    def compute_reward(self, action: MyEnvV4Action, correct: str):
        reward = 0.0

        if action.decision.lower() == correct:
            reward += 0.7

        if len(action.reasoning) > 10:
            reward += 0.3

        return min(reward, 1.0)

    async def step(self, action: MyEnvV4Action):
        self.step_count += 1
        task = self.tasks[self.current_task]

        reward = self.compute_reward(action, task["correct"])

        done = True  # one-step task

        return StepResult(
            observation=Observation(
                request_id=task["id"],
                employee_role=task["role"],
                item=task["item"],
                cost=task["cost"],
                policy=task["policy"]
            ),
            reward=reward,
            done=done
        )

    async def close(self):
        pass
