from pprint import pprint
from textwrap import dedent
from typing import List

import ray
from pydantic import Field

from agent_actors.agent import Agent
from agent_actors.chains.child import Check, Do


class ChildAgent(Agent):
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

    def run(self, task: str, working_memory: List[ray.ObjectRef] = []):
        try:
            self.status = "running"

            for _ in range(self.max_iterations):
                self.task = task

                context = self.get_context()
                if any(working_memory):
                    context += "\n" + "\n\n".join(ray.get(working_memory))

                result = self.do(
                    inputs=dict(context=context, task=self.task),
                    return_only_outputs=True,
                )

                for action, action_result in result["intermediate_steps"]:
                    learning = f"Thought: {action.log}\n{action_result}"
                    self.add_memory(learning)

                learning = f"Task: {self.task}\nResult:\n{result['output']}"
                self.add_memory(learning)

                task = self.check.run(context=context, learning=learning)
                if "Complete" in task:
                    return learning

            return learning
        finally:
            self.task = ""
            self.status = "idle"
