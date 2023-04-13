from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Prioritize(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return cls(
            prompt=PromptTemplate(
                template=dedent(
                    """\
                    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}

                    Consider the ultimate objective of your team: {objective}
                    Also consider the dependencies within the tasks.

                    Do not remove any tasks. Return the result as a numbered list, like:
                    #. First task
                    #. Second task

                    Start the task list with number {next_task_id}.
                    """
                ),
                input_variables=["task_names", "next_task_id", "objective"],
            ),
            llm=llm,
            verbose=verbose,
        )
