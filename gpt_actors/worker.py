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

    def call(
        self, *, objective: str, task: Dict[str, Any], dependencies: List[ray.ObjectRef]
    ):
        context = self.get_summary()
        if any(dependencies):
            dependencies = ray.get(dependencies)

        iterations_count = 0
        next_task = TaskRecord(**task)
        while next_task and iterations_count < self.max_iterations:
            result = self.do.run(objective=objective, task=task, context=context)
            next_task = self._check_task_result(
                objective=objective, task=task, result=result
            )

        self.add_memory(result)
        return result

    def _generate_next_task(self, objective: str, task: TaskRecord, result: str):
        result = self.check.run(
            objective=objective,
            task=task.description,
            result=result,
        ).strip()

        if result.startswith("Complete"):
            print("Objective complete!")
        elif result.startswith("Next"):
            print("Task complete!")
        else:
            return TaskRecord(**task, description=result)
