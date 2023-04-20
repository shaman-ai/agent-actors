from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Plan(LLMChain):
    """Plan the next step."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(
            prompt=PromptTemplate(
                template=dedent(
                    """\
                    You are an expert planner that plans the next steps for an AI agent.
                    The agent's objective is: {objective}

                    These are the incomplete tasks, in order: {incomplete_tasks}

                    Decide the minimal set of new tasks for the agent to complete that do not overlap with the incomplete tasks.

                    Return the result as a numbered list, like:
                    #. First task
                    #. Second task

                    If no additional tasks items are required to complete the objective, don't return anything.
                    """
                ),
                input_variables=["objective", "incomplete_tasks"],
            ),
            llm=llm,
            verbose=verbose,
        )
