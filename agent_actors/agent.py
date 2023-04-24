import re
from datetime import datetime
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import ray
from langchain.agents import Tool
from langchain.chat_models.base import BaseChatModel
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from pydantic import BaseModel, Field

from agent_actors.chains.agent import (GenerateInsights, PrioritizeMemory,
                                       ReflectionTopics, WorkingMemory)


def _parse_list(text: str) -> List[str]:
    """Parse a newline-separated string into a list of strings."""
    return [
        re.sub(r"^\s*\d+\.\s*", "", line).strip()
        for line in re.split(r"\n", text.strip())
    ]


@ray.remote
class Actor:
    agent: "Agent"

    def __init__(self, agent: "Agent"):
        self.agent = agent

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)

    def call(self, method_name, *args, **kwargs):
        method = getattr(self.agent, method_name)
        return method(*args, **kwargs)


class Agent(BaseModel):
    # Configuration
    name: str
    traits: List[str] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    objective: str = Field(default="")

    llm: BaseChatModel = Field(init=False)
    memory: TimeWeightedVectorStoreRetriever = Field(init=False)

    reflect_every: int = 3
    reflect_importance_trigger: Optional[float] = float("inf")
    cumulative_importance: float = 0.5

    verbose: bool = False
    max_tokens_limit: int = 1200

    # State
    status: str = "idle"
    working_memory: str = ""
    memories_since_last_reflection: int = 0
    last_refreshed: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = dedent(
            """\
            Should {agent_name} react to the observation, and if so, what would be an appropriate reaction? Respond in one line. If the action is to engage in dialogue, write:
            SAY: "what to say"
            otherwise, write:
            REACT: {agent_name}'s reaction (if anything).
            Either do nothing, react, or say something but not both.

            """
        )

        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        self.add_memory(f"{self.name} observed {observation} and reacted by {result}")
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = 'What would {agent_name} say? To end the conversation, write: GOODBYE: "what to say". Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split("\n")[0]
        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            self.add_memory(f"{self.name} observed {observation} and said {farewell}")
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            self.add_memory(
                f"{self.name} observed {observation} and said {response_text}"
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

    def fetch_memories(self, observation: str) -> List[Document]:
        return self.memory.get_relevant_documents(observation)

    def is_time_to_reflect(self) -> bool:
        return (
            self.memories_since_last_reflection >= self.reflect_every - 1
            or self.cumulative_importance > self.reflect_importance_trigger
        )

    def _add_memory(self, memory_content: str) -> List[str]:
        importance_score = self._score_cumulative_importance(memory_content)
        print(importance_score)

        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )

        self.memories_since_last_reflection += 1
        self.cumulative_importance += importance_score

        return self.memory.add_documents([document])

    def add_memory(self, memory_content: str) -> List[str]:
        result = self._add_memory(memory_content)
        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if self.is_time_to_reflect():
            self.pause_to_reflect()
            self.memories_since_last_reflection = 0
            self.cumulative_importance = 0.0
        return result

    def get_header(self):
        return dedent(
            f"""\
            Name: {self.name}
            Traits: {", ".join(self.traits)}
            Objective: {self.objective}
            """
        )

    def get_context(self, force_refresh: bool = False) -> str:
        if not self.working_memory or force_refresh or self.is_time_to_reflect():
            self._compute_agent_working_memory()

        return f"{self.get_header()}Working Memory:\n{self.working_memory}"

    def _compute_agent_working_memory(self):
        relevant_memories = self.fetch_memories(
            f"{self.name}'s memories about {self.objective}"
        )
        if not any(relevant_memories):
            self.working_memory = "Nothing"
        else:
            working_memory_chain = WorkingMemory.from_llm(
                llm=self.llm, verbose=self.verbose
            )

            self.working_memory = working_memory_chain.run(
                context=self.get_header(),
                objective={self.objective},
                relevant_memories="\n".join(
                    f"{m.page_content}" for m in relevant_memories
                ),
            ).strip()

        self.last_refreshed = datetime.utcnow()
        self.memories_since_last_reflection = 0

        return self.working_memory

    def _get_topics_of_reflection(
        self, last_k: Optional[int] = None
    ) -> Tuple[str, str, str]:
        if last_k is None:
            last_k = self.reflect_every
        observations = self.memory.memory_stream[-last_k:]
        reflection_topics_chain = ReflectionTopics(llm=self.llm, verbose=self.verbose)
        return _parse_list(
            reflection_topics_chain.run(
                context=self.get_header(),
                objective=self.objective,
                memories="\n".join(o.page_content for o in observations),
            ).strip()
        )

    def _get_insights_on_topic(self, topic: str) -> List[str]:
        related_memories = self.fetch_memories(topic)
        if not any(related_memories):
            return []

        generate_insights_chain = GenerateInsights.from_llm(
            llm=self.llm, verbose=self.verbose
        )
        return _parse_list(
            generate_insights_chain.run(
                context=self.get_header(),
                topic=topic,
                related_statements="\n".join(
                    f"{i+1}. {memory.page_content}"
                    for i, memory in enumerate(related_memories)
                ),
            ).strip()
        )

    def pause_to_reflect(self):
        if self.status == "reflecting":
            return []

        old_status = self.status
        self.status = "reflecting"
        insights = self._pause_to_reflect()
        self.status = old_status
        return insights

    def _pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic(topic)
            for insight in insights:
                self._add_memory(insight)
            new_insights.extend(insights)
        self.memories_since_last_reflection += len(new_insights)
        if self.memories_since_last_reflection >= self.reflect_every - 1:
            self._pause_to_reflect()
        return new_insights

    def _score_cumulative_importance(
        self, memory_content: str, weight: float = 0.15
    ) -> float:
        """Score the absolute importance of the given memory."""
        score = (
            PrioritizeMemory.from_llm(llm=self.llm, verbose=self.verbose)
            .run(memory_content=memory_content, objective=self.objective)
            .strip()
        )
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(score[0]) / 10) * weight

    def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content not in content_strs:
                content_strs.add(mem.page_content)
                created_time = mem.metadata["created_at"].strftime(
                    "%B %d, %Y, %I:%M %p"
                )
                content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc.page_content)
        return "; ".join(result[::-1])
