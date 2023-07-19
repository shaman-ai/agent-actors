from typing import List

import ray
from dotenv import load_dotenv
from faiss import IndexFlatL2
from langchain.agents import Tool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseRetriever
from langchain.vectorstores import FAISS

from agent_actors.callback_manager import ConsolePrettyPrintManager
from agent_actors.child import ChildAgent
from agent_actors.parent import ParentAgent
from agent_actors.toolkit import default_toolkit


class TestSystem:
    llm: BaseChatModel
    long_term_memory: BaseRetriever
    callback_manager: BaseCallbackManager
    tools: List[Tool] = []

    @classmethod
    def setup_class(cls):
        load_dotenv()
        cls.callback_manager = ConsolePrettyPrintManager([])
        cls.llm = ChatOpenAI(
            temperature=0.25,
            model_name="gpt-3.5-turbo",
            callback_manager=cls.callback_manager,
            max_tokens=1024,
        )
        cls.tools = default_toolkit()
        cls.long_term_memory = TimeWeightedVectorStoreRetriever(
            vectorstore=FAISS(
                embedding_function=OpenAIEmbeddings().embed_query,
                index=IndexFlatL2(1536),
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        )
        ray.init()

    @classmethod
    def teardown_class(cls):
        ray.shutdown()

    def create_child(
        self, name: str, traits: List[str], max_iterations=3
    ) -> ChildAgent:
        return ChildAgent(
            name=name,
            traits=traits,
            max_iterations=max_iterations,
            llm=self.llm,
            long_term_memory=self.long_term_memory,
            tools=self.tools,
            callback_manager=self.callback_manager,
        )

    def create_parent(
        self, name: str, traits: List[str], max_iterations=1, **kwargs
    ) -> ParentAgent:
        return ParentAgent(
            **kwargs,
            name=name,
            traits=traits,
            max_iterations=max_iterations,
            llm=self.llm,
            # tools=self.tools,
            long_term_memory=self.long_term_memory,
            callback_manager=self.callback_manager,
        )

    def test_child_ability(self):
        amir = self.create_child(
            name="Amir",
            traits=["smart", "kind"],
            max_iterations=3,
        )
        amir.run(task="What is Sergey Brin's age multiplied by 12?")

    def test_parent_overhead(self):
        luke = self.create_parent(
            name="Luke",
            traits=["project manager", "golden retriever energy"],
            max_iterations=3,
        )
        amir = self.create_child(
            name="Amir",
            traits=["smart", "kind"],
            max_iterations=3,
        )
        luke.children = {0: amir}
        luke.run(task="What is Sergey Brin's age multiplied by 12?")

    def test_parallel_map_reduce(self):
        jiang = self.create_child(
            name="Jiang",
            traits=["sharp", "math, stats, and data whiz", "capitalist"],
        )
        sophia = self.create_child(
            name="Sophia",
            traits=["deep thinker", "contrarian", "empathetic"],
        )
        esther = self.create_child(
            name="Esther",
            traits=["great writer"],
        )
        jerry = self.create_child(
            name="Jerry",
            traits=["funny", "creative", "comedian", "executive assistant"],
        )
        luke = self.create_parent(
            name="Luke",
            traits=["AI project manager", "golden retriever energy"],
            max_iterations=3,
            children={0: jiang, 1: sophia, 2: esther, 42: jerry},
        )
        luke.run(
            task="I need an executive report on Artificial General Intelligence and a list of 5 relevant jokes and quotes."
        )

    def test_nested_parent_tree(self):
        jiang = self.create_child(
            name="Jiang",
            traits=["sharp", "math, stats, and data whiz", "capitalist"],
        )
        sophia = self.create_child(
            name="Sophia",
            traits=["deep thinker", "contrarian", "empathetic"],
        )
        esther = self.create_child(
            name="Esther",
            traits=["great writer"],
        )
        luke = self.create_parent(
            name="Luke",
            traits=["AI project manager", "golden retriever energy"],
            max_iterations=3,
            children={0: jiang, 1: sophia, 2: esther},
        )
        amir = self.create_child(
            name="Amir",
            traits=["funny", "creative", "kind"],
        )
        cyrus = self.create_parent(
            "Cyrus",
            traits=["kind human"],
            max_iterations=2,
            children={0: luke, 42: amir},
        )

        cyrus.run(
            task="I need an executive report on Artificial General Intelligence and a list of 5 related jokes and quotes."
        )

    def test_writing_a_technical_blog_post(self):
        writer = self.create_child(
            "Writer",
            traits=[
                "brilliant technical writer",
                "programmer in a past life",
            ],
        )
        researcher = self.create_child(
            "Researcher",
            traits=[
                "expert researcher",
                "expert programmer",
            ],
        )
        cto = self.create_parent(
            "CTO",
            traits=["kind human", "expert programmer", "visionary technologist"],
            children={2: researcher, 42: writer},
        )

        with open("./README.md", "r") as f:
            """
            You can load your actors with memories, here, since all actors share
            the same vectorstore, this will be available to all actors.
            """
            cto.add_memory(f.read())

        cto.run(
            task=f"Write a blog post about Agent Actors, a new python repository that helps you build trees of AI agents that work together to solve more complex problems. Use your memory rather than searches to learn more about Agent Actors. If you're going to search, then search for the Actor Model of Concurrency, Elixir / OTP and Plan-Do-Check-Act cycles"
        )
