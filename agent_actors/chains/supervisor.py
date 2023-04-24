import json
from textwrap import dedent
from typing import Any, Dict, List

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel


class Plan(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate(
                template_format="jinja2",
                template=dedent(
                    """\
                    {{ context }}

                    Your objective is: {{ objective }}

                    This is your team:
                    {{ worker_summary }}

                    Decide and assign the minimal set of new tasks required for your workers to complete that do not overlap with the incomplete tasks. The same task should not be assigned to multiple workers. Workers can have multiple tasks assigned to them. Do not assign tasks to workers that don't exist. Task dependencies must refer to existing tasks. Task IDs should be unique across all tasks and start at 0. When referencing other tasks, use the format [worker #.task #]. Return just the JSON array, in the following format, starting with [ and ending with ].

                    ```
                    [
                        {
                            "task_id": <task id>,
                            "worker_id": <worker id>,
                            "objective": <task objective>,
                            "dependencies": [{
                                "worker_id": <worker id>,
                                "task_id": <task id>
                            }]
                        },
                        ...
                    ]
                    ```
                    """
                ),
                input_variables=[
                    "context",
                    "objective",
                    "worker_summary",
                ],
            ),
        )


class Adjust(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                template_format="jinja2",
                template=dedent(
                    """\
                    {{ context }}

                    You are an continuous-improvement AI that reviews tasks completed by agents and decides what to do next.

                    The objective of these tasks was: {{ objective }}

                    The results were:
                    {{ results }}

                    Based on these results, imagine your confidence of having completed the objective as a number between 1 and 10. Return just the JSON in the following format:

                    ```
                    {"confidence": confidence, "speak": "<what to say to your copilot>"}
                    ```
                    """
                ),
            ),
        )

    @property
    def output_keys(self) -> List[str]:
        return ["confidence", "speak"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(super()._call(inputs)[self.output_key].strip())