from textwrap import dedent

from langchain import LLMChain, PromptTemplate

from agent_actors.chains.base import JSONChain


class Plan(JSONChain):
    @classmethod
    def from_llm(cls, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            prompt=PromptTemplate(
                template_format="jinja2",
                template=dedent(
                    """\
                    {{ context }}

                    Your task is: {{ task }}

                    Here is your team:
                    {{ child_summary }}

                    Decide and assign the minimal set of new sub-tasks for your team members complete. Assign sub-tasks to who is best suited for the sub-task based on their traits and working memory. Team members can have multiple sub-tasks assigned to them. Do not duplicate work. Do not assign tasks to team members that don't exist.

                    Dependencies must refer to existing sub-tasks. When referencing other tasks, use the format [worker #.task #]. Return just the JSON array, in the following format, starting with [ and ending with ].

                    Good luck!


                    ```
                    [
                        {
                            "task_id": <incrementing int starting at 0 per child>,
                            "child_id": <assigned team member id>,
                            "task": <task task>,
                            "dependencies": [{
                                "child_id": <assigned team member id>,
                                "task_id": <int>
                            }]
                        },
                        ...
                    ]
                    ```
                    """
                ),
                input_variables=[
                    "context",
                    "task",
                    "child_summary",
                ],
            ),
        )


class Adjust(JSONChain):
    @classmethod
    def from_llm(cls, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            prompt=PromptTemplate.from_template(
                template_format="jinja2",
                template=dedent(
                    """\
                    {{ context }}

                    You are an continuous-improvement AI that reviews tasks completed by agents and decides what to do next.

                    The task of these tasks was: {{ task }}

                    The results were:
                    {{ results }}

                    Based on these results, imagine your confidence of having completed the task as a number between 1 and 10. Return just the JSON in the following format:

                    ```
                    {
                        "confidence": confidence,
                        "speak": "<what to say to your copilot>",
                        "result": <synthesized result satisfying objective>
                    }
                    ```
                    """
                ),
            ),
        )
