from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel


class Plan(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
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

                    Decide the minimal set of new tasks for your workers to complete that do not overlap with the incomplete tasks.

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

                    If no additional tasks items are required to complete the objective, then don't return anything. Task IDs should be unique across all tasks and start at 1. Return just the JSON array, starting with [ and ending with ].
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
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You are an continuous-improvement AI that reviews tasks completed by agents and decides what to do next.

                    The objective of these tasks was: {objective}

                    The results were:
                    {results}

                    Imagine your confidence of having reasonably fulfilled the objective as a number between 1 and 10.

                    Based on these results, if your confidence is greater than 7, then return:

                    Complete: <summary of results>

                    Otherwise, propose a better objective.
                    """
                ),
            ),
        )
