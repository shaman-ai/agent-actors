import json
from typing import Any, Dict, List

import ray
from langchain.chat_models.base import BaseChatModel
from pydantic import Field

from gpt_actors.agent import Agent, AgentActor
from gpt_actors.chains.supervisor import Adjust, Plan
from gpt_actors.models import AgentRecord, TaskRecord
from gpt_actors.worker import WorkerAgent


def print_task_list(tasks_by_worker: List[Dict[str, Any]]):
    for worker in tasks_by_worker:
        agent = AgentRecord(**worker)
        print(agent)
        for task in worker["tasks"]:
            print(TaskRecord(**task, worker_id=agent.worker_id))


class SupervisorAgent(Agent):
    plan: Plan = Field(init=False)
    adjust: Adjust = Field(init=False)
    worker_llm: BaseChatModel = Field(init=False)

    def __init__(self, *args, **kwargs):
        llm = kwargs["llm"]
        super().__init__(
            *args, **kwargs, plan=Plan.from_llm(llm), adjust=Adjust.from_llm(llm)
        )

    def run(self, *args, objective: str, **kwargs):
        self.status = "running"
        context = self.get_summary()
        print("Context", context)

        tasks_by_worker = json.loads(
            self.plan.run(*args, **kwargs, objective=objective, context=context).strip()
        )

        print_task_list(tasks_by_worker)

        task_records = []
        workers = {}

        for worker in tasks_by_worker:
            agent = AgentRecord(**worker)
            task_records.extend(
                (TaskRecord(**t, worker_id=agent.worker_id) for t in worker["tasks"])
            )
            workers[agent.worker_id] = AgentActor.remote(
                agent=WorkerAgent(
                    name=agent.name,
                    traits=", ".join(agent.traits),
                    llm=self.worker_llm,
                    memory=self.memory,
                )
            )

        tasks = {}
        for t in task_records:
            tasks[t.id] = workers[t.worker_id].run.remote(
                objective=objective,
                task=dict(t),
                dependencies=[tasks[d.id] for d in t.dependencies],
            )

        tasks_in_progress = list(tasks.values())
        while any(tasks_in_progress):
            tasks_completed, tasks_in_progress = ray.wait(
                tasks_in_progress,
                num_returns=min(self.reflect_every, len(tasks_in_progress)),
            )

            results = ray.get(tasks_completed)

            for memory in results:
                self._add_memory(memory)

            self.pause_to_reflect()

        self.status = "idle"
        return self.get_summary()
