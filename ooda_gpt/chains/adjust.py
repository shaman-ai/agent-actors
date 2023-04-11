from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM


class Adjust(LLMChain):
    """
    The phase where a process is improved. Records from the "do" and "check" phases help identify issues with the process. These issues may include problems, non-conformities, opportunities for improvement, inefficiencies, and other issues that result in outcomes that are evidently less-than-optimal. Root causes of such issues are investigated, found, and eliminated by modifying the process. Risk is re-evaluated. At the end of the actions in this phase, the process has better instructions, standards, or goals. Planning for the next cycle can proceed with a better baseline. Work in the next do phase should not create a recurrence of the identified issues; if it does, then the action was not effective.
    """
