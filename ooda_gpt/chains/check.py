from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Check(LLMChain):
    """
    During the check phase, the data and results gathered from the do phase are evaluated. Data is compared to the expected outcomes to see any similarities and differences. The testing process is also evaluated to see if there were any changes from the original test created during the planning phase.
    """
