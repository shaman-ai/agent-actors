from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Check(LLMChain):
    """
    During the check phase, the data and results gathered from the do phase are evaluated. Data is compared to the expected outcomes to see any similarities and differences. The testing process is also evaluated to see if there were any changes from the original test created during the planning phase.
    """

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
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
