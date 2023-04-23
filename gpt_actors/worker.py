from textwrap import dedent
from typing import Any, Dict, List, Optional

import ray
from langchain import LLMChain, PromptTemplate
from langchain.agents import MRKLChain, Tool, ZeroShotAgent
from langchain.chains import LLMMathChain
from langchain.chains.base import Chain
from langchain.chat_models.base import BaseChatModel
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field

from gpt_actors.models import TaskRecord
from gpt_actors.utilities.log import print_heading


@ray.remote
class Worker:
    def __init__(self, chain: Chain):
        self.chain = chain

    def call(self, *args, **kwargs):
        return self.chain.run(*args, **kwargs)


class WorkerChain(Chain, BaseModel):
    llm: BaseChatModel = Field(init=False)
    do: "Do" = Field(init=False)
    check: "Check" = Field(init=False)
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
    ) -> "WorkerChain":
        return cls(
            llm=llm,
            do=Do.from_llm(llm, verbose=verbose),
            check=Check.from_llm(llm, verbose=verbose),
            vectorstore=vectorstore,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        return ["objective", "task", "dependencies"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        next_task = TaskRecord(**inputs["task"])
        dependencies = inputs["dependencies"]
        iteration_count = 0
        result = ""

        print_heading("WORKER STARTING", color="cyan")
        if dependencies:
            dependencies = ray.get(dependencies)

        while next_task and iteration_count < self.max_iterations:
            iteration_count += 1
            print(next_task)
            result = self._perform(
                objective=objective,
                task=next_task,
                context=dependencies,
            )
            print_heading("TASK RESULT", color="grey")
            print(result)
            next_task = self._verify(objective=objective, task=next_task, result=result)

        return {"result": result}

    def _perform(
        self, objective: str, task: TaskRecord, context: List[str], k: int = 5
    ) -> str:
        # Get related, completed tasks. In the future, this should be the global memory.
        completed_tasks = self.vectorstore.similarity_search_with_score(objective, k=k)
        if completed_tasks:
            completed_tasks, _ = zip(
                *sorted(completed_tasks, key=lambda x: x[1], reverse=True)
            )
            completed_tasks = [
                str(item.metadata["description"]) for item in completed_tasks
            ]
        else:
            completed_tasks = []

        # Get the result of the task
        result = self.do.run(objective=objective, task=task, context=completed_tasks)

        # Add the result to the vectorstore
        self.vectorstore.add_texts(
            texts=[result],
            metadatas=[dict(task)],
            ids=[f"result_{task.actor_id}_{task.id}"],
        )

        return result

    def _verify(self, objective: str, task: TaskRecord, result: str):
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


class Do(MRKLChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        tools = [
            WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper()),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
            Tool(
                name="Calculator",
                func=LLMMathChain(llm=llm).run,
                description="useful for when you need to do math. Input should be a math expression.",
            ),
            Tool(
                name="Search",
                func=GoogleSerperAPIWrapper().run,
                description="useful for when you need to answer questions using up-to-date information from the internet. Input should be a search query.",
            ),
            Tool(
                name="TODO",
                func=LLMChain(
                    llm=llm,
                    prompt=PromptTemplate.from_template(
                        "You are a data-driven planner who is an expert at coming up with a todo list for a given objective. Come up with the minimal todo list to complete the objective: {objective}."
                    ),
                ).run,
                description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
            ),
        ]

        prefix = dedent(
            """\
            You are an AI who performs one task based on the following objective: {objective}

            Take into account these previously completed tasks: {context}
            """
        )
        suffix = dedent(
            """\
            Question: {task}

            Scratchpad: {agent_scratchpad}
            """
        )

        agent = ZeroShotAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["objective", "task", "context", "agent_scratchpad"],
        )

        return cls(agent=agent, tools=tools, verbose=verbose)


class Check(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            prompt=PromptTemplate(
                template=dedent(
                    """\
                    You are an evaluation AI that checks the result of an AI agent against the objective: {objective}

                    Review the results of the last completed task below and decide whether the task was accurately accomplishes the objective or the task.

                    The task was: {task}
                    The result was: {result}

                    If the result accomplishes the objective, return "Complete".
                    If the result accomplishes the task, return "Next".
                    Otherwise, return just the description of a task that will produce the expected result, with no preamble.
                    """
                ),
                input_variables=["objective", "task", "result"],
            ),
            llm=llm,
            verbose=verbose,
        )


WorkerChain.update_forward_refs()
