from textwrap import dedent

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel


class Summarize(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    How would you summarize the following history:

                    {related_memories}

                    Do not embellish.

                    Summary:
                    """
                )
            ),
        )


class Importance(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True) -> LLMChain:
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    On a scale of 1 to 10, where 1 is purely irerelevant and 10 is relevant to the objective "{objective}", rate the likely relevance of the following piece of memory. Respond with a single integer.

                    Memory: {memory_content}
                    Rating:
                    """
                )
            ),
        )


class Insights(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = False):
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    Statements about {topic}:
                    {related_statements}

                    What are 5 high-level insights we can infer from the above statements? Provide each insight on a new line. (example format: insight (because of 1, 5, 3))
                    """
                )
            ),
        )


class ObservedEntity(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = False):
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            + "\nEntity="
        )
        return cls(llm=llm, prompt=prompt, verbose=verbose)


class EntityAction(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = False):
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            + "\nThe {entity} is"
        )
        return cls(llm=llm, prompt=prompt, verbose=verbose)
