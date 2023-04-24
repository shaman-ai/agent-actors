from pprint import pprint
from textwrap import dedent
from typing import Any, Dict, List

import ray
from pydantic import Field

from agent_actors.agent import Agent
from agent_actors.chains.worker import Check, Do


class WorkerAgent(Agent):
    max_iterations: int = 3
    do: Do = Field(init=False)
    check: Check = Field(init=False)

    def __init__(self, *args, **kwargs):
        chain_params = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", False),
            callback_manager=kwargs.get("callback_manager", None),
        )

        super().__init__(
            *args,
            **kwargs,
            do=Do.from_llm(**chain_params, tools=kwargs["tools"]),
            check=Check.from_llm(**chain_params),
        )

    def run(self, objective: str, working_memory: List[ray.ObjectRef]):
        try:
            self.status = "running"

            for _ in range(self.max_iterations):
                self.objective = objective

                context = self.get_context()
                if any(working_memory):
                    context += "\n" + "\n\n".join(ray.get(working_memory))

                result = self.do(
                    inputs=dict(context=context, objective=self.objective),
                    return_only_outputs=True,
                )

                if result["intermediate_steps"]:
                    print("=== INTERMEDIATE STEPS ===")
                    pprint(result["intermediate_steps"])

                learning = f"""[[MEMORY]]\nObjective: {self.objective}\nResult:\n{result["output"]}"""
                self.add_memory(learning)

                objective = self.check.run(context=context, learning=learning)
                if "Complete" in objective:
                    return learning

            return learning
        finally:
            self.objective = ""
            self.status = "idle"
