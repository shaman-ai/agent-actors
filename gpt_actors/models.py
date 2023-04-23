from typing import List

from pydantic import BaseModel, Field


class TaskRef(BaseModel):
    worker_id: int = Field(...)
    task_id: int = Field(...)

    @property
    def id(self) -> str:
        return f"{self.worker_id}.{self.task_id}"


class TaskRecord(TaskRef):
    description: str = Field(...)
    dependencies: List[TaskRef] = Field(default_factory=list)

    def __str__(self) -> str:
        fmt_task = f"[{self.id}] {self.description}"
        fmt_deps = (
            f"""(depends on {', '.join(f"[{d.id}]" for d in self.dependencies)})"""
            if any(self.dependencies)
            else ""
        )
        return f"{fmt_task} {fmt_deps}"


class AgentRecord(BaseModel):
    worker_id: int
    name: str
    traits: List[str]

    def __str__(self) -> str:
        return f"[Worker {self.worker_id}] {self.name} ({self.traits})"
