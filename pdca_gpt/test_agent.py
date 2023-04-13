from faiss import IndexFlatL2
from langchain import OpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from pdca_gpt.agent import Agent


class TestAgent:
    agent: Agent

    @classmethod
    def setup_class(cls):
        cls.agent = Agent.from_llm(
            llm=OpenAI(temperature=0),
            vectorstore=FAISS(
                embedding_function=OpenAIEmbeddings().embed_query,
                index=IndexFlatL2(1536),
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            ),
            verbose=True,
            max_iterations=5,
        )

    def test_logic(self):
        result = self.agent.run(
            objective="What is Sergey Brin's age raised to the power of 4?"
        )
        assert result == "5764801"

    def test_thinking(self):
        self.agent.run(objective="How can we ensure the safe development of AGI?")
