from textwrap import dedent

import ray
from langchain import LLMChain, PromptTemplate
from langchain.agents import MRKLChain, Tool, ZeroShotAgent
from langchain.chains import LLMMathChain
from langchain.chat_models.base import BaseChatModel
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)


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
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
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
            ),
        )
