from textwrap import dedent
from typing import Any, Dict, List

import ray
from pydantic import Field

from agent_actors.agent import Agent
from agent_actors.chains.worker import TaskAgent
from agent_actors.models import TaskRecord


class WorkerAgent(Agent):
    task_agent: TaskAgent = Field(init=False)

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            task_agent=TaskAgent.from_llm(
                llm=kwargs["llm"],
                tools=kwargs.get("tools", []),
                verbose=kwargs.get("verbose", False),
            ),
        )

    def run(self, objective: str, working_memory: List[ray.ObjectRef]):
        try:
            self.status = "running"
            self.objective = objective

            context = self.get_context()
            if any(working_memory):
                context += "\n" + "\n".join(ray.get(working_memory))

            result = self.task_agent.run(
                context=context,
                objective=self.objective,
            )

            learning = dedent(
                f"""\
                Objective Description: {self.objective}
                Objective Result: {result}
                """
            )
            self.add_memory(learning)

            return learning
        finally:
            self.objective = ""
            self.status = "idle"
