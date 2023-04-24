import ray


@ray.remote(max_restarts=3, max_task_retries=3)
class AgentActor:
    agent: "Agent"  # type: ignore due to circular dependency

    def __init__(self, agent):
        self.agent = agent

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)

    def call(self, method_name, *args, **kwargs):
        method = getattr(self.agent, method_name)
        return method(*args, **kwargs)
