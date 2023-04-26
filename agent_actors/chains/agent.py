import re
from textwrap import dedent
from typing import Any, Dict, List

from langchain import LLMChain, PromptTemplate
from langchain.chat_models.base import BaseChatModel


class NaturalLanguageListChain(LLMChain):
    output_key = "items"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, List[str]]:
        text = super()._call(inputs)[self.output_key].strip()
        items = [
            re.sub(r"^\s*\d+\.\s*", "", line).strip()
            for line in re.split(r"\n", text.strip())
        ]
        return {"items": items}


class WorkingMemory(NaturalLanguageListChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You have been tasked with {task}.

                    Here is a list of your relevant memories:
                    {relevant_memories}

                    Your task is to synthesize from these memories a list of 1 to 12 insights and knowledge that will serve as your working memory for accomplishing the task.

                    Working Memory: \
                    """
                )
            ),
        )


class MemoryStrength(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    On a scale of 1 to 10, where 1 is purely irrelevant and 10 is salient, rate the likely relevance of the following result to the task. Respond with a single integer.

                    {memory_content}
                    Relevance: \
                    """
                )
            ),
        )


class Synthesis(NaturalLanguageListChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You have been tasked with {task}.

                    Here is a list of your recent memories:
                    {memories}

                    Synthesize your memories for long-term storage, with 1 synthesis per line.
                    """
                )
            ),
        )


class GenerateInsights(NaturalLanguageListChain):
    @classmethod
    def from_llm(cls, llm: BaseChatModel, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {context}

                    You are to generate insights about {topic}.

                    Here are the relevant memories:
                    {related_statements}

                    Generate 3 high-level insights we can infer from the above statements. Provide each insight on a new line. Format in the following format:

                    Insight: <insight text> [memory source 1, ...]
                    """
                )
            ),
        )
