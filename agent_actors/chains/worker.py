from textwrap import dedent
from typing import List

from langchain import LLMChain, PromptTemplate
from langchain.agents import MRKLChain, Tool, ZeroShotAgent
from langchain.chat_models.base import BaseChatModel


class TaskAgent(MRKLChain):
    @classmethod
    def from_llm(
        cls, llm: BaseChatModel, tools: List[Tool], verbose: bool = True
    ) -> LLMChain:
        agent = ZeroShotAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            prefix="Complete the task below. You have access to the following tools:",
            format_instructions=dedent(
                """\
                Use the following format:

                Objective: the input task you must complete
                Thought: you should always think about what to do
                Reasoning: the reasoning behind your thought
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                Reflection: constructive self-criticism
                ... (this Thought/Reasoning/Action/Action Input/Observation/Reflection can repeat N times)
                Thought: The task has been completed appropriately and accurately
                Final Answer: the final result of this task
            """
            ),
            suffix=dedent(
                """\
                {context}

                Objective: {objective}
                Thought: {agent_scratchpad}\
                """
            ),
            input_variables=[
                "context",
                "objective",
                "agent_scratchpad",
            ],
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

                    Here is the relevant context: {context}

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
