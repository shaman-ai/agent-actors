from typing import Any, Dict, List, Optional

import ray
from pydantic import Field

from gpt_actors.agent import Agent
from gpt_actors.chains.worker import Check, Do
from gpt_actors.models import TaskRecord


class WorkerAgent(Agent):
    max_iterations: Optional[int] = Field(default=2)
    do: Do = Field(init=False)
    check: Check = Field(init=False)

    def __init__(self, *args, **kwargs):
        llm = kwargs["llm"]
        super().__init__(
            *args, **kwargs, do=Do.from_llm(llm), check=Check.from_llm(llm)
        )

    def run(
        self, *, objective: str, task: Dict[str, Any], dependencies: List[ray.ObjectRef]
    ):
        self.status = "running"
        context = self.get_summary()
        if any(dependencies):
            dependencies = ray.get(dependencies)

        iterations_count = 0
        next_task = TaskRecord(**task)
        while next_task and iterations_count < self.max_iterations:
            print(f"Doing {next_task.description} with context {context}")
            result = self.do.run(
                objective=objective,
                task_description=next_task.description,
                context=context,
            )
            next_task = self._generate_next_task(
                objective=objective, task=next_task, result=result, context=context
            )

        self.add_memory(result)
        self.status = "idle"
        return result

    def _generate_next_task(
        self, objective: str, task: TaskRecord, result: str, context: str
    ):
        result = self.check.run(
            context=context,
            objective=objective,
            result=result,
            task=task.description,
        ).strip()

        if result.startswith("Complete"):
            print("Objective complete!")
        elif result.startswith("Next"):
            print("Task complete!")
        else:
            return TaskRecord(**task, description=result)
