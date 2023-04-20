from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.agents import MRKLChain, Tool, ZeroShotAgent
from langchain.chains import LLMMathChain
from langchain.llms import BaseLLM
from langchain.tools.human.tool import HumanInputRun
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)


class Do(MRKLChain):
    """MRKL chain to for doing an action."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        tools = [
            WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper()),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
            HumanInputRun(),
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
