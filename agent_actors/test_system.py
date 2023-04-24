import ray
from dotenv import load_dotenv
from faiss import IndexFlatL2
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.tools.wolfram_alpha.tool import WolframAlphaQueryRun
from langchain.utilities import (
    GoogleSerperAPIWrapper,
    WikipediaAPIWrapper,
    WolframAlphaAPIWrapper,
)
from langchain.vectorstores import FAISS

from agent_actors.agent import Actor
from agent_actors.supervisor import SupervisorAgent
from agent_actors.worker import WorkerAgent


class TestSystem:
    llm: BaseChatModel
    memory: TimeWeightedVectorStoreRetriever
    supervisor: Actor

    @classmethod
    def setup_class(cls):
        load_dotenv()
        ray.init()

        cls.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        cls.memory = TimeWeightedVectorStoreRetriever(
            vectorstore=FAISS(
                embedding_function=OpenAIEmbeddings().embed_query,
                index=IndexFlatL2(1536),
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        )

        wolfram_alpha_tool = WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper())
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

        tools = [
            Tool(
                name="Wolfram Alpha",
                func=wolfram_alpha_tool.run,
                description=wolfram_alpha_tool.description,
            ),
            Tool(
                name="Wikipedia",
                func=wikipedia_tool.run,
                description=wikipedia_tool.description,
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
                name="TODO",
                func=LLMChain(
                    llm=cls.llm,
                    prompt=PromptTemplate.from_template(
                        "You are a data-driven planner who is an expert at coming up with a todo list for a given objective. Come up with the minimal todo list to complete the objective: {objective}."
                    ),
                ).run,
                description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
            ),
        ]

        workers = [
            WorkerAgent(
                name="José Juarez",
                traits=[
                    "Expert AI practioner working at OpenAI on existential AGI risk",
                    "effective altruist",
                    "revolutionary",
                ],
                llm=cls.llm,
                memory=cls.memory,
                tools=tools,
                verbose=True,
            ),
            WorkerAgent(
                name="Anastasia Rand",
                traits=[
                    "Expert AI practioner working at DeepMind on existential AGI risk",
                    "effective altruist",
                    "individualist",
                ],
                llm=cls.llm,
                memory=cls.memory,
                tools=tools,
                verbose=True,
            ),
        ]

        cls.supervisor = Actor.remote(
            agent=SupervisorAgent(
                name="Alireza",
                traits=["expert project manager"],
                llm=cls.llm,
                memory=cls.memory,
                workers=workers,
                verbose=True,
                max_iterations=1,
            )
        )

    @classmethod
    def teardown_class(cls):
        ray.shutdown()

    def test_simple_search_and_math(self):
        ref = self.supervisor.run.remote(
            objective="What is Sergey Brin's age times 12?"
        )
        result = ray.get(ref)
        assert "588" in result

    def test_research(self):
        ray.get(
            self.supervisor.run.remote(
                objective="Who are the founders of OpenAI and what are their roles and ages?"
            )
        )

    def test_research_and_thinking(self):
        ray.get(
            self.supervisor.run.remote(
                objective="How can we ensure the safe development of AGI?",
            )
        )
