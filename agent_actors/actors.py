import ray


@ray.remote(max_restarts=3, max_task_retries=3)
class AgentActor:
    agent: "Agent"  # type: ignore due to circular dependency

    def __init__(self, agent):
        self.agent = agent

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)

    def call(self, method: str = "", *args, **kwargs):
        method = self.agent if not method else getattr(self.agent, method)
        return method(*args, **kwargs)


@ray.remote(max_restarts=3, max_task_retries=3)
class ChainActor:
    chain: "Chain"  # type: ignore due to circular dependency

    def __init__(self, chain):
        self.chain = chain

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)

    def call(self, method: str = "", *args, **kwargs):
        method = self.chain if not method else getattr(self.chain, method)
        return method(*args, **kwargs)
