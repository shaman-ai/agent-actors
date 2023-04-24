from datetime import datetime
from pprint import pprint
from typing import Any, Dict, List

import ray
from dotenv import load_dotenv
from faiss import IndexFlatL2
from langchain.agents import Tool
from langchain.callbacks import CallbackManager, StdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)
from langchain.vectorstores import FAISS

from agent_actors.agent import Actor
from agent_actors.supervisor import SupervisorAgent
from agent_actors.worker import WorkerAgent


class ConsolePrettyPrinter(StdOutCallbackHandler):
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        super().on_chain_end(outputs, **kwargs)
        pprint(outputs)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        super().on_chain_start(serialized, inputs, **kwargs)
        pprint(inputs)


class TestSystem:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    long_term_memory = TimeWeightedVectorStoreRetriever(
        vectorstore=FAISS(
            embedding_function=OpenAIEmbeddings().embed_query,
            index=IndexFlatL2(1536),
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
    )
    callback_manager = CallbackManager(
        handlers=[
            # StreamlitCallbackHandler(),
            ConsolePrettyPrinter(),
        ]
    )
    tools: List[Tool]
    supervisor_agent: SupervisorAgent
    supervisor_actor: Actor

    def test_single_lookup_and_math(self):
        # One worker
        self.supervisor_agent.set_workers(self.workers[:1])
        self.supervisor_agent.run(objective="What is Sergey Brin's age times 12?")

    def test_multi_lookup(self):
        self.supervisor_agent.set_workers(self.workers)
        self.supervisor_agent.run(
            objective="Generate a markdown table of the top 5 most populous cities in the world, along with their population and GDP."
        )

    def test_critical_thinking(self):
        self.supervisor_agent.set_workers(self.workers)
        self.supervisor_agent.run(
            objective="Develop a thorough plan to ensure the safe development of AGI.",
        )

    def test_strategic_thinking(self):
        workers = [
            WorkerAgent(
                name="José Juarez",
                traits=[
                    "chief operating officer",
                ],
                llm=self.llm,
                long_term_memory=self.long_term_memory,
                tools=self.tools,
                callback_manager=self.callback_manager,
            ),
            WorkerAgent(
                name="Anastasia Rand",
                traits=[
                    "sales lead",
                ],
                llm=self.llm,
                long_term_memory=self.long_term_memory,
                tools=self.tools,
                callback_manager=self.callback_manager,
            ),
            WorkerAgent(
                name="Justina Jackson",
                traits=[
                    "digital marketer",
                    "website developer",
                ],
                llm=self.llm,
                long_term_memory=self.long_term_memory,
                tools=self.tools,
                callback_manager=self.callback_manager,
            ),
            WorkerAgent(
                name="Jay Ferris",
                traits=["creative", "strategist"],
                llm=self.llm,
                long_term_memory=self.long_term_memory,
                tools=self.tools,
                callback_manager=self.callback_manager,
            ),
        ]

        self.supervisor_agent.set_workers(workers)
        self.supervisor_agent.run(
            objective="Develop a thorough go-to-market plan for a digital agency that provides marketing and web development services for climate fintech companies.",
        )

    @classmethod
    def setup_class(cls):
        load_dotenv()
        ray.init()

        cls.tools = [
            Tool(
                name="Wolfram Alpha",
                func=WolframAlphaAPIWrapper().run,
                description="A wrapper around Wolfram Alpha. Useful for when you need to answer questions about Math or Science. Input should be a search query.",
            ),
            Tool(
                name="Wikipedia",
                func=WikipediaAPIWrapper().run,
                description="A wrapper around Wikipedia that fetches page summaries. Useful when you need a summary of a person, place, company, historical event, or other subject. Input is typically a noun, like a person, place, company, historical event, or other subject.",
            ),
            Tool(
                name="Calculator",
                func=LLMMathChain(llm=cls.llm).run,
                description="useful for when you need to do math. Input should be a math expression.",
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
        cls.workers = [
            WorkerAgent(
                name="José Juarez",
                traits=[
                    "effective altruist",
                ],
                llm=cls.llm,
                long_term_memory=cls.long_term_memory,
                tools=cls.tools,
                callback_manager=cls.callback_manager,
            ),
            WorkerAgent(
                name="Anastasia Rand",
                traits=[
                    "individualist",
                ],
                llm=cls.llm,
                long_term_memory=cls.long_term_memory,
                tools=cls.tools,
                callback_manager=cls.callback_manager,
            ),
        ]
        cls.supervisor_agent = SupervisorAgent(
            name="Alireza",
            traits=["expert strategist"],
            llm=cls.llm,
            long_term_memory=cls.long_term_memory,
            max_iterations=1,
            callback_manager=cls.callback_manager,
        )

    @classmethod
    def teardown_class(cls):
        ray.shutdown()
