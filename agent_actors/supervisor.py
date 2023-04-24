import json
from collections import defaultdict
from pprint import pprint
from typing import Dict, List

import ray
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field

from agent_actors.agent import Actor, Agent
from agent_actors.chains.supervisor import Adjust, Plan
from agent_actors.models import TaskRecord


class SupervisorAgent(Agent):
    plan: Plan = Field(init=False)
    adjust: Adjust = Field(init=False)
    workers: Dict[int, Agent] = Field(init=False, default_factory=dict)
    children: Dict[int, ray.ObjectRef] = Field(init=False, default_factory=dict)

    def __init__(self, *args, **kwargs):
        chain_params = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", True),
            callback_manager=kwargs.get("callback_manager", None),
        )

        super().__init__(
            *args,
            **kwargs,
            plan=Plan.from_llm(**chain_params),
            adjust=Adjust.from_llm(**chain_params),
        )

    def set_workers(self, workers: List[Agent]):
        self.workers = {}
        self.children = {}
        for id, agent in enumerate(workers):
            self.workers[id] = agent
            self.children[id] = Actor.remote(agent)

    def run_in_child(self, worker_id: int, *args, **kwargs):
        return self.children[worker_id].run(*args, **kwargs)

    def run(self, objective: str):
        try:
            self.status = "running"
            self.objective = objective

            planned_tasks = [
                TaskRecord(**t)
                for t in json.loads(
                    self.plan.run(
                        context=self.get_context(),
                        objective=self.objective,
                        worker_summary="\n\n".join(
                            f"ID: {id}\n{worker.get_context()}"
                            for id, worker in self.workers.items()
                        ),
                    ).strip()
                )
            ]

            if self.verbose:
                worker_tasks = defaultdict(list)
                for t in planned_tasks:
                    worker_tasks[t.worker_id].append(t)
                for worker_id, worker_tasks in worker_tasks.items():
                    print("\n=== WORKER TASKS ===")
                    print(self.workers[worker_id].get_context())
                    print("Tasks:")
                    for t in worker_tasks:
                        print(t)

            task_futures = {}
            for t in planned_tasks:
                task_futures[t.id] = self.children[t.worker_id].run.remote(
                    objective=t.objective,
                    working_memory=[task_futures[d.id] for d in t.dependencies],
                )

            task_results = []
            tasks_in_progress = list(task_futures.values())
            while any(tasks_in_progress):
                tasks_completed, tasks_in_progress = ray.wait(
                    tasks_in_progress,
                    num_returns=min(self.reflect_every, len(tasks_in_progress)),
                )

                results = ray.get(tasks_completed)

                for result in results:
                    task_results.append(result)

            self.pause_to_reflect()

            reflection = self.adjust(
                inputs=dict(
                    context=self.get_context(),
                    objective=self.objective,
                    results="\n".join(task_results),
                )
            )

            if reflection["confidence"] >= 8:
                return AgentFinish(reflection, "success")

            return AgentAction("Human", "What should my next objective be?", "info")
        finally:
            self.objective = ""
            self.status = "idle"
