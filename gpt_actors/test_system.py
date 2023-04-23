import ray
from dotenv import load_dotenv
from faiss import IndexFlatL2
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS

from gpt_actors.actor import Actor
from gpt_actors.supervisor import SupervisorAgent


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
        cls.supervisor = Actor.remote(
            agent=SupervisorAgent(
                name="AI Agent Supervisor",
                traits="expert at planning research tasks",
                memory=cls.memory,
                worker_llm=cls.llm,
                verbose=True,
                llm=cls.llm,
            )
        )

    def test_search_and_math(self):
        ref = self.supervisor.call.remote(
            objective="What is Sergey Brin's age times 12?"
        )
        result = ray.get(ref)
        assert "588" in result

    def test_thinking(self):
        ray.get(
            self.supervisor.call.remote(
                objective="How can we ensure the safe development of AGI?",
            )
        )
