import os

from collections import deque
from typing import Dict, List, Optional, Any

from faiss import IndexFlatL2
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chains.base import Chain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.vectorstores import FAISS, VectorStore
from pydantic import BaseModel, Field

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


