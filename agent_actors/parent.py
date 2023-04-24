from collections import defaultdict
from pprint import pprint
from typing import List

import ray
from langchain.schema import AgentAction, AgentFinish
from pydantic import Field

from agent_actors.agent import Agent
from agent_actors.chains.parent import Adjust, Plan
from agent_actors.child import ChildAgent
from agent_actors.models import TaskRecord


class ParentAgent(Agent):
    plan: Plan = Field(init=False)
    adjust: Adjust = Field(init=False)

    def __init__(self, *args, **kwargs):
        chain_params = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", True),
            callback_manager=kwargs.get("callback_manager", None),
        )

        super().__init__(
            *args,
            **kwargs,
            plan=Plan.from_llm(**chain_params),
            adjust=Adjust.from_llm(**chain_params),
        )

    def run(self, task: str, working_memory: List[ray.ObjectRef] = []):
        try:
            self.status = "running"
            self.task = task

            context = self.get_context() + "\n".join(ray.get(working_memory))

            planned_tasks = [
                TaskRecord(**t)
                for t in self.plan(
                    inputs=dict(
                        context=context,
                        task=self.task,
                        child_summary="\n\n".join(
                            f"ID: {id}\n{child.get_context()}"
                            for id, child in self.children.items()
                        ),
                    ),
                )["json"]
            ]

            if self.verbose:
                child_tasks = defaultdict(list)
                for sub_task in planned_tasks:
                    child_tasks[sub_task.child_id].append(sub_task)
                for child_id, child_tasks in child_tasks.items():
                    print(f"\n=== CHILD {child_id} TASKS ===")
                    pprint(child_tasks)

            task_result_refs = {}

            for sub_task in planned_tasks:
                if self.verbose:
                    print(f"\n\n\n=== CHILD TASK {sub_task} ===")

                child_id = sub_task.child_id

                if child_id not in self.children:
                    self.add_child(
                        ChildAgent(
                            llm=self.llm,
                            verbose=self.verbose,
                            name=f"Team Member {sub_task.child_id}",
                            traits=["focused", "team player"],
                            max_iterations=3,
                            callback_manager=self.callback_manager,
                        )
                    )

                task_result_refs[sub_task.id] = self.children[
                    child_id
                ].actor.run.remote(
                    task=sub_task.task,
                    working_memory=[
                        task_result_refs[d.id] for d in sub_task.dependencies
                    ],
                )

            task_results = []
            tasks_in_progress = list(task_result_refs.values())
            while any(tasks_in_progress):
                tasks_completed, tasks_in_progress = ray.wait(
                    tasks_in_progress,
                    num_returns=min(self.reflect_every, len(tasks_in_progress)),
                )

                results = ray.get(tasks_completed)

                for result in results:
                    task_results.append(result)

            self.pause_to_reflect()

            reflection = self.adjust.run(
                context=self.get_context(),
                task=self.task,
                results="\n".join(task_results),
            )

            if reflection["confidence"] >= 8:
                return AgentFinish(reflection, "success")

            return AgentAction("Human", "What should my next task be?", "info")
        finally:
            self.task = ""
            self.status = "idle"
