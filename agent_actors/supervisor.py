import json
from collections import defaultdict
from typing import Any, Dict, List

import ray
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field

from agent_actors.agent import Actor, Agent
from agent_actors.chains.supervisor import Adjust, Plan
from agent_actors.models import TaskRecord


def print_task_list(workers: Dict[any, Agent], tasks: List[TaskRecord]):
    worker_tasks = defaultdict(list)
    for t in tasks:
        worker_tasks[t.worker_id].append(t)
    for worker_id, tasks in worker_tasks.items():
        print(workers[worker_id].get_context())
        for t in tasks:
            print(t)


class SupervisorAgent(Agent):
    plan: Plan = Field(init=False)
    adjust: Adjust = Field(init=False)
    workers: Dict[int, Agent] = Field(init=False, default_factory=dict)
    children: Dict[int, ray.ObjectRef] = Field(init=False, default_factory=dict)
    max_iterations: int = Field(default=1)

    def __init__(self, *args, workers: List[Agent], **kwargs):
        llm = kwargs["llm"]
        verbose = kwargs.get("verbose", False)

        super().__init__(
            *args,
            **kwargs,
            plan=Plan.from_llm(llm, verbose=verbose),
            adjust=Adjust.from_llm(llm, verbose=verbose),
            workers={id: agent for id, agent in enumerate(workers)},
        )

        for id, agent in self.workers.items():
            self.children[id] = Actor.remote(agent)

    def run_in_child(self, worker_id: int, *args, **kwargs):
        return self.children[worker_id].run(*args, **kwargs)

    def run(self, objective: str):
        try:
            self.status = "running"
            self.objective = objective

            for _ in range(self.max_iterations):
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

                print_task_list(self.workers, planned_tasks)

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
                        task_results.push(result)
                        self._add_memory(result)

                    self.pause_to_reflect()

                reflection = self.adjust.run(
                    context=self.get_context(),
                    objective=self.objective,
                    results="\n".join(task_results),
                )

                if "Complete" in reflection:
                    return AgentFinish(
                        {"summary": reflection, "results": task_results}, "success"
                    )

                self.objective = reflection

        finally:
            self.objective = ""
            self.status = "idle"
