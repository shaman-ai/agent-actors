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
                    You are an expert planner that plans the next steps for a group of AI workers, as modeled by the actor model of concurrency.

                    The overall objective is: {{ objective }}

                    Decide the minimal set of new tasks for the workers to complete that do not overlap with the incomplete tasks. A single worker will perform tasks in the order they are given, but multiple workers can work on different tasks in parallel, waiting on results from other workers. Apply a topological sort to the tasks to ensure that the tasks are completed in the correct order. If a task is dependent on another task, it should probably be done by the same worker.

                    Return the result as a JSON list in the following format:

                    ```
                    [
                        {
                            "worker_id": <worker id>,
                            "name": <worker name>,
                            "traits": [<traits suitable for this worker to possess that would make it successful in its role>, ...],
                            "tasks": [
                                {
                                    "task_id": <task id>,
                                    "description": <description>,
                                    "dependencies": [{
                                        "worker_id": <worker id>,
                                        "task_id": <task id>
                                    }]
                                },
                                ...
                            ]
                        }
                        ...
                    ]
                    ```

                    If no additional tasks items are required to complete the objective, then don't return anything. Task IDs should be sequential for a given worker, and start at 1. Agents should be numbered sequentially, starting at 1. Return just the JSON array, starting with [ and ending with ].
                    """
                ),
                input_variables=["objective"],
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
                    You are an evaluation AI that adjusts the result of an AI supervisor against the objective: {objective}

                    The results were: {results}

                    If the result accomplishes the objective, return "COMPLETE". Otherwise, return "REDO".
                    """
                ),
            ),
        )
