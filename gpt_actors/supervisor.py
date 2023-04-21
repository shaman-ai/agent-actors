import json
from collections import defaultdict, deque
from textwrap import dedent
from typing import Any, Dict, List, Optional

import ray
from langchain import LLMChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field

from gpt_actors.models import TaskRecord
from gpt_actors.utilities.log import print_heading
from gpt_actors.worker import Worker, WorkerChain


def print_task_list(ts: List[TaskRecord]):
    tasks_by_actor = defaultdict(list)
    for t in ts:
        tasks_by_actor[t.actor_id].append(t)
    for actor_id, tasks in tasks_by_actor.items():
        print(f"Agent {actor_id}")
        for t in tasks:
            print(t)


@ray.remote
class Supervisor:
    def __init__(self, chain: Chain):
        self.chain = chain

    def call(self, *args, **kwargs):
        return self.chain.run(*args, **kwargs)


class SupervisorChain(Chain, BaseModel):
    llm: BaseChatModel = Field(init=False)
    plan: "Plan" = Field(init=False)
    review: "Review" = Field(init=False)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = Field(default=2)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_llm(
        cls,
        llm: BaseChatModel,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs,
    ) -> "SupervisorChain":
        return cls(
            llm=llm,
            plan=Plan.from_llm(llm, verbose=verbose),
            review=Review.from_llm(llm, verbose=verbose),
            vectorstore=vectorstore,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        iteration_count = 0

        while iteration_count < self.max_iterations:
            print_heading("SUPERVISOR PLANNING", color="cyan")
            plan = self.plan.run(objective=objective)
            tasks = deque(TaskRecord(**attrs) for attrs in json.loads(plan))
            print_task_list(tasks)

            workers = {
                id: Worker.options(
                    name=f"worker_{id}", namespace=objective, get_if_exists=True
                ).remote(
                    WorkerChain.from_llm(llm=self.llm, vectorstore=self.vectorstore)
                )
                for id in set(t.actor_id for t in tasks)
            }

            task_refs = {}
            for t in tasks:
                task_refs[t.id] = workers[t.actor_id].call.remote(
                    objective=objective,
                    task=dict(t),
                    dependencies=[task_refs[d.id] for d in t.dependencies],
                )

            results = ray.get(list(task_refs.values()))
            print(results)

            print_heading("SUPERVISOR REVIEW", color="cyan")
            if "COMPLETE" in self.review.run(objective=objective, results=results):
                return {"result": results}


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
                    You are an expert planner that plans the next steps for a group of AI agents, as modeled by the actor model of concurrency.

                    The overall objective is: {{ objective }}

                    Decide the minimal set of new tasks for the agents to complete that do not overlap with the incomplete tasks. A single agent will perform tasks in the order they are given, but multiple agents can work on different tasks in parallel, waiting on results from other agents. Apply a topological sort to the tasks to ensure that the tasks are completed in the correct order. If a task is dependent on another task, it should probably be done by the same agent.

                    Return the result as a JSON list in the following format:

                    ```
                    [
                        {
                            "actor_id": <actor id>,
                            "task_id": <task id>,
                            "description": <description>,
                            "dependencies": [{
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
                input_variables=["objective"],
            ),
        )


class Review(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            prompt=PromptTemplate(
                template=dedent(
                    """\
                    You are an evaluation AI that reviews the result of an AI supervisor against the objective: {objective}

                    The results were: {results}

                    If the result accomplishes the objective, return "COMPLETE". Otherwise, return "REDO".
                    """
                ),
                input_variables=["objective", "results"],
            ),
            llm=llm,
            verbose=verbose,
        )


SupervisorChain.update_forward_refs()
