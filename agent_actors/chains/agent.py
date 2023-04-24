from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel


class WorkingMemory(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You have been tasked with {objective}.

                    Here is a list of your relevant memories:
                    {relevant_memories}

                    Your task synthesize from these memories a list of insights and knowledge that will serve as your working memory for accomplishing the objective.

                    Working Memory: \
                    """
                )
            ),
        )


class PrioritizeMemory(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    On a scale of 1 to 10, where 1 is purely irrelevant and 10 is salient, rate the likely relevance of the following piece of memory to the objective "{objective}". Respond with a single integer.

                    Memory: {memory_content}
                    Rating: \
                    """
                )
            ),
        )


class ReflectionTopics(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You have been tasked with {objective}.

                    Here is a list of your recent memories:
                    {memories}

                    Given only the memories above, generate 3 questions to ask to synthesize new knowledge or create insights. Provide each question on a new line.
                    """
                )
            ),
        )


class GenerateInsights(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = False):
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You are to generate insights about {topic}.

                    Here are the relevant memories:
                    {related_statements}

                    What are 3 high-level insights we can infer from the above statements? Provide each insight on a new line.

                    Format in the following format: <insight> [statement #, ...]
                    """
                )
            ),
        )
