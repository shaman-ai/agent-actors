from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Plan(LLMChain):
    """Plan the next step."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(
            prompt=PromptTemplate(
                template_format="jinja2",
                template=dedent(
                    """\
                    You are an expert planner that plans the next steps for a group of AI agents, as modeled by the actor model of concurrency.

                    The overall objective is: {{ objective }}

                    These are the incomplete tasks, in order: {{ incomplete_tasks }}

                    Decide the minimal set of new tasks for the agents to complete that do not overlap with the incomplete tasks. A single agent should perform tasks in the order they are given, but multiple agents can work on different tasks in parallel, waiting on results from other agents. Group the tasks in such a way that multiple agents can complete it in parallel. Apply a topological sort to the tasks to ensure that the tasks are completed in the correct order.

                    Return the result as a JSON list in the following format:

                    ```
                    [
                        {
                            "actor_id": <actor id>,
                            "task_id": <task id>,
                            "task_name": <task name>,
                            "depends_on": [{
                                "actor_id": <actor id>,
                                "task_id": <task id>
                            }]
                        },
                        ...
                    ]
                    ```

                    If no additional tasks items are required to complete the objective, then don't return anything. Task IDs should be sequential for a given agent, and start at 1. Agents should be numbered sequentially, starting at 1. Return just the JSON array, starting with [ and ending with ].
                    """
                ),
                input_variables=["objective", "incomplete_tasks"],
            ),
            llm=llm,
            verbose=verbose,
        )
