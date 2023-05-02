import re
from datetime import datetime
from textwrap import dedent
from typing import Dict, List, Optional, Tuple

import ray
from langchain import LLMChain
from langchain.agents import Tool
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field

from agent_actors.actors import AgentActor
from agent_actors.chains.agent import (
    GenerateInsights,
    MemoryStrength,
    Synthesis,
    WorkingMemory,
)


class Agent(BaseModel):
    # Configuration
    name: str
    traits: List[str] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    task: str = Field(default="")
    children: Dict[int, "Agent"] = Field(default_factory=dict)

    reflect_every: int = 10
    reflect_strength_trigger: Optional[float] = float("inf")
    cumulative_strength: float = 0.5

    verbose: bool = False
    max_tokens_limit: int = 1200

    # State
    status: str = "idle"
    working_memory_state: str = ""
    memories_since_last_reflection: int = 0
    last_refreshed: datetime = Field(default_factory=datetime.now)

    # Internal
    llm: BaseChatModel = Field(init=False)
    long_term_memory: BaseRetriever = Field(init=False)
    working_memory: LLMChain = Field(init=False)
    synthesis: LLMChain = Field(init=False)
    insights_generator: LLMChain = Field(init=False)
    memory_strength: LLMChain = Field(init=False)
    actor: ray.ObjectRef = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        chain_params = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", True),
            callback_manager=kwargs.get("callback_manager", None),
        )
        super().__init__(
            *args,
            **kwargs,
            working_memory=WorkingMemory.from_llm(**chain_params),
            synthesis=Synthesis.from_llm(**chain_params),
            insights_generator=GenerateInsights.from_llm(**chain_params),
            memory_strength=MemoryStrength.from_llm(**chain_params),
        )
        self.actor = AgentActor.remote(self)

    def set_children(self, children: Dict[int, "Agent"]):
        self.children = children

    def add_child(self, child_agent: "Agent"):
        self.children[len(self.children)] = child_agent

    def remove_child(self, agent_id: int):
        return self.children.pop(agent_id)

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
        return self.long_term_memory.get_relevant_documents(observation)

    def is_time_to_reflect(self) -> bool:
        return (
            self.memories_since_last_reflection >= self.reflect_every - 1
            or self.cumulative_strength > self.reflect_strength_trigger
        )

    def _add_memory(self, memory_content: str) -> List[str]:
        datum = "[[MEMORY]]\n" + memory_content
        strength_score = self._predict_memory_strength(datum)

        nodes = self.long_term_memory.add_documents(
            [Document(page_content=datum, metadata={"strength": strength_score})]
        )

        if self.verbose:
            print(datum if len(datum) < 280 else datum[:280] + "...")

        self.memories_since_last_reflection += 1
        self.cumulative_strength += strength_score

        return nodes

    def add_memory(self, memory_content: str) -> List[str]:
        result = self._add_memory(memory_content)
        # After an agent has processed a certain amount of memories (as measured by
        # aggregate strength), it is time to reflect on recent events to add
        # more synthesized memories to the agent's memory stream.
        if self.is_time_to_reflect():
            self.pause_to_reflect()
        return result

    def get_header(self):
        return dedent(
            f"""\
            Name: {self.name}
            Traits: {", ".join(self.traits)}
            Task: {self.task}
            """
        )

    def get_context(self, force_refresh: bool = False) -> str:
        context = self.working_memory_state
        if not context or force_refresh or self.is_time_to_reflect():
            context = self._compute_agent_working_memory()

        return f"{self.get_header()}Working Memory:\n{context}\n"

    def _compute_agent_working_memory(self):
        if not self.task:
            return "[Empty]"

        relevant_memories = self.fetch_memories(self.task)
        if not any(relevant_memories):
            return "[Empty]"

        self.working_memory_state = "\n".join(
            f"{n}. {mem}"
            for (n, mem) in enumerate(
                self.working_memory.run(
                    context=self.get_header(),
                    task=self.task,
                    relevant_memories="\n".join(
                        f"{m.page_content}" for m in relevant_memories
                    ),
                )
            )
        )

        self.last_refreshed = datetime.utcnow()
        self.memories_since_last_reflection = 0

        return self.working_memory_state

    def _synthesize_memories(
        self, last_k: Optional[int] = None
    ) -> Tuple[str, str, str]:
        if last_k is None:
            last_k = self.reflect_every

        observations = self.long_term_memory.memory_stream[-last_k:]
        if not any(observations):
            return []

        return self.synthesis.run(
            context=self.get_header(),
            task=self.task,
            memories="\n".join(o.page_content for o in observations),
        )

    def generate_insights(self, topic: str) -> List[str]:
        related_memories = self.fetch_memories(topic)
        if not any(related_memories):
            return []

        insights = self.insights_generator.run(
            context=self.get_header(),
            topic=topic,
            related_statements="\n".join(
                f"{i+1}. {memory.page_content}"
                for i, memory in enumerate(related_memories)
            ),
        )

        for insight in insights:
            self._add_memory(insight)

        return insights

    def pause_to_reflect(self):
        if self.status == "reflecting":
            return []

        old_status = self.status
        self.status = "reflecting"
        insights = self._pause_to_reflect()
        self.cumulative_strength = 0.0
        self.status = old_status
        return insights

    def _pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        new_insights = self._synthesize_memories()
        for insight in new_insights:
            self.add_memory(insight)
        self.memories_since_last_reflection = 0
        return new_insights

    def _predict_memory_strength(self, memory_content: str) -> float:
        """Score the absolute strength of the given memory."""
        score = self.memory_strength.run(memory_content=memory_content).strip()
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return float(score[0]) / 10

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
        for doc in self.long_term_memory.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc.page_content)
        return "; ".join(result[::-1])
