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
                    You are an expert planner that uses the result of an AI agent to decide new todo items with the following objective: {objective}.

                    The last todo was: {task}
                    The result was: {result}

                    These are incomplete todo items: {incomplete_tasks}

                    Based on the result, decide the minimal set of new todo items to be completed by the AI system that do not overlap with incomplete todo items. Return the tasks as an array. If no additional todo items are required, then don't return anything.
                    """
                ),
                input_variables=["objective", "task", "result", "incomplete_tasks"],
            ),
            llm=llm,
            verbose=verbose,
        )
