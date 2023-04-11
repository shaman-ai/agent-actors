from langchain import OpenAI, GoogleSerperAPIWrapper, LLMChain, PromptTemplate
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.llms import BaseLLM

todo_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    ),
)

search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    ),
]


class Act(ZeroShotAgent):
    """MRKL chain to act."""

    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        return super().from_llm(
            llm,
            prefix=cls.prefix,
            suffix=cls.suffix,
            input_variables=["objective", "task", "context", "agent_scratchpad"],
            verbose=verbose,
        )
