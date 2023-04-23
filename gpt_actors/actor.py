import ray

from gpt_actors.agent import Agent


@ray.remote
class Actor:
    agent: Agent

    def __init__(self, agent: Agent):
        self.agent = agent

    def call(self, *args, **kwargs):
        return self.agent.call(*args, **kwargs)
