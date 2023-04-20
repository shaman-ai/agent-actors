from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.schema import AgentFinish
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field

from pdca_gpt.chains import Adjust, Check, Do, Plan

load_dotenv()


def print_heading(heading: str, color: str):
    return print(f"\033[{color}m\033[1m\n*****{heading}*****\n\033[0m\033[0m")


def print_task_list(task_list: List[Dict]):
    print_heading("TASK LIST", color="95")
    for t in task_list:
        print(str(t["task_id"]) + ": " + t["task_name"])


def print_next_task(task: Dict):
    print_heading("NEXT TASK", color="92")
    print(str(task["task_id"]) + ": " + task["task_name"])


def print_refined_task(task: Dict):
    print_heading("REFINED TASK", color="94")
    print(str(task["task_id"]) + ": " + task["task_name"])


def print_task_result(result: str):
    print_heading("TASK RESULT", color="93")
    print(result)


class Agent(Chain, BaseModel):
    task_list: deque = Field(default_factory=deque)
    plan: Plan = Field(...)
    do: Do = Field(...)
    check: Check = Field(...)
    adjust: Adjust = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    def plan_tasks(self, task: str, objective: str):
        generated_tasks = self.plan.run(
            objective=objective,
            task=task,
            incomplete_tasks=", ".join(t["task_name"] for t in self.task_list),
        ).split("\n")

        for t in map(str.strip, generated_tasks):
            if t:
                self.task_id_counter += 1
                self.add_task({"task_name": t, "task_id": self.task_id_counter})

    def adjust_tasks(self, objective: str, current_task_id: int):
        adjusted_tasks = self.adjust.run(
            objective=objective,
            task_names=[t["task_name"] for t in self.task_list],
            next_task_id=current_task_id + 1,
        ).split("\n")

        self.task_list = deque()
        for t in map(str.strip, adjusted_tasks):
            if t:
                task_parts = t.split(".", 1)
                if len(task_parts) == 2:
                    task_id = int(task_parts[0].strip())
                    task_name = task_parts[1].strip()
                    self.task_list.append({"task_id": task_id, "task_name": task_name})

    def do_task(self, objective: str, task: Dict, k: int = 5) -> str:
        # Get related, completed tasks
        completed_tasks = self.vectorstore.similarity_search_with_score(objective, k=k)
        if completed_tasks:
            completed_tasks, _ = zip(
                *sorted(completed_tasks, key=lambda x: x[1], reverse=True)
            )
            completed_tasks = [str(item.metadata["task"]) for item in completed_tasks]
        else:
            completed_tasks = []

        # Get the result of the task
        result = self.do.run(
            objective=objective, context=completed_tasks, task=task["task_name"]
        )

        # Add the result to the vectorstore
        result_id = f"result_{task['task_id']}"
        self.vectorstore.add_texts(
            texts=[result],
            metadatas=[{"task": task["task_name"]}],
            ids=[result_id],
        )
        return result

    def check_task(self, objective: str, task: Dict, result: str):
        result = self.check.run(
            objective=objective,
            task=task["task_name"],
            result=result,
        ).strip()

        if result.startswith("Complete"):
            print("Objective complete!")
            return AgentFinish({"state": "complete"}, "")
        elif result.startswith("Next"):
            print("Task complete!")
            return AgentFinish({"state": "next"}, "")

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]

        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 0, "task_name": first_task})
        num_iters = 0

        # Step 4: Create new tasks

        while self.task_list:
            task = self.task_list.popleft()
            self.plan_tasks(
                task=task["task_name"],
                objective=objective,
            )

            print_task_list(self.task_list)
            # Step 1: Pull the first task
            print_next_task(task)

            # Step 2: Do the task
            result = self.do_task(objective=objective, task=task)
            print_task_result(result)

            # Step 3: Check the result
            # ok = self.check_task(objective=objective, task=task, result=result)
            # if not ok:
            #     # If the task is not complete, refine it and add it to the top of the task list
            #     refined_task = {"task_id": task["task_id"], "task_name": result}
            #     print_refined_task(refined_task)
            #     self.task_list.insert(0, refined_task)
            # elif ok.return_values["state"] == "complete":
            #     return {"result": result}



            # Step 5: Prioritize tasks
            self.adjust_tasks(objective=objective, current_task_id=task["task_id"])

            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                print_heading("AGENT ENDING", color="91")
                break

        return {"result": result}

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "Agent":
        return cls(
            plan=Plan.from_llm(llm, verbose=verbose),
            do=Do.from_llm(llm, verbose=verbose),
            check=Check.from_llm(llm, verbose=verbose),
            adjust=Adjust.from_llm(llm, verbose=verbose),
            vectorstore=vectorstore,
            **kwargs,
        )
