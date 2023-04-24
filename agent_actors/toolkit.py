from datetime import datetime
from typing import List

from langchain import LLMChain, OpenAI
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)


def default_toolkit() -> List[Tool]:
    return [
        Tool(
            name="Generative LLM",
            func=OpenAI(temperature=0.75).__call__,
            description="A generative large-language-model, can be used to generate text in response to a prompt. Input should be a prompt.",
        ),
        Tool(
            name="Wolfram Alpha",
            func=WolframAlphaAPIWrapper().run,
            description="A wrapper around Wolfram Alpha. Useful for when you need to do math, statistical analysis, or answer questions about science. Input should be a search query.",
        ),
        Tool(
            name="Wikipedia",
            func=WikipediaAPIWrapper().run,
            description="A wrapper around Wikipedia that fetches page summaries. Useful when you need a summary of a person, place, company, historical event, or other subject. Input is typically a noun, like a person, place, company, historical event, or other subject.",
        ),
        Tool(
            name="Search",
            func=GoogleSerperAPIWrapper().run,
            description="useful for when you need to answer questions using up-to-date information from the internet. Input should be a search query.",
        ),
        Tool(
            name="DateTime",
            func=lambda _: datetime.utcnow().isoformat(),
            description="Useful for when you want to know the current date and time. Input is ignored.",
        ),
    ]
