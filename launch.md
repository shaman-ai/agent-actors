---
Title: Unleashing the Power of AI Collaboration with Parallelized LLM Agent Actor Trees
Subtitle: Discover a novel approach to AI agent collaboration using the Actor Model of Concurrency.
---

## Introduction

The field of AI has seen significant advancements in recent years, with AI agents now capable of tackling complex tasks. Despite these advances, it has remained a challenge to effectively parallelize and coordinate the efforts of multiple AI agents working together. Introducing Agent Actors - a groundbreaking solution that enables developers to create and manage trees of AI agents that collaborate on complex tasks using the Actor Model of Concurrency.

In this blog post, we'll explore the Actor Model of Concurrency, key features of Agent Actors, the possibilities it opens up, and how you can start building your own agent trees. We hope to inspire the LLM community to experiment with new architectures for self-referencing GPTs.

## The Actor Model of Concurrency

The Actor Model of Concurrency is a powerful paradigm for managing concurrent computation, making it an ideal foundation for Agent Actors. In the Actor Model, an "actor" is an independent computational entity that communicates with other actors through asynchronous message-passing. Each actor can perform tasks, create new actors, and send messages in response to incoming messages. This model offers several advantages for building stateful AI agents with parallelism:

1. **Isolation**: Actors are self-contained, encapsulating their own state, which prevents unwanted data sharing or race conditions. This isolation ensures that each agent can work independently without affecting others, making it easier to build and maintain parallel AI systems.
2. **Asynchronous Communication**: Actors communicate through message-passing, allowing them to work concurrently without the need for explicit synchronization. This enables more efficient use of system resources and improved scalability.
3. **Fault Tolerance**: The Actor Model allows for better fault isolation and recovery, as errors in one actor do not automatically propagate to others. This enhances the overall robustness of the system, which is particularly important when managing multiple AI agents working together.

## Key Features and Benefits

Agent Actors boasts an array of powerful features that make it possible to build and manage collaborative AI agents:

- **Time Weighted Long-Term Memory**: Agent Actors utilizes `langchain.retrievers.TimeWeightedVectoreStoreRetriever` to implement time-weighted long-term memory, allowing agents to access relevant information with ease. This feature helps improve the quality of agent output by ensuring that the most pertinent data is always available.
- **Synthesized Working Memory**: Agents draw insights from their memories and synthesize them into a working memory of 1-12 items for use with zero-shot prompts. By maintaining a compact working memory, agents can focus on the most important information, leading to more accurate and relevant results.
- **Plan-Do-Check-Adjust (PDCA) Framework**: Agent Actors implements the PDCA framework for continuous improvement, enabling agents to work more effectively over time. By iteratively refining their performance, agents can produce increasingly better output as they learn from experience, and perform a sort of gradient descent towards task completion.
- **Automatic Planning and Task Distribution**: The `ParentAgent` class plans tasks for its children agents and distributes them for parallel execution. This feature streamlines the coordination of agent activities, ensuring that tasks are optimally distributed for maximum efficiency.
- **Parallel Execution of Agents**: `ChildAgent`s work in parallel to execute tasks and check results, ensuring efficient use of resources and faster results. By allowing agents to work concurrently, large problems can be solved faster since independent threads can process in parallel.
- **Customizable AI Agent Trees**: Developers can create their own trees of autonomous AI agents by nesting `ParentAgent`s or combining them with `ChildAgent`s. This flexibility enables developers to create tailored solutions that best suit their specific needs.

## Exciting Possibilities

Agent Actors unlocks a new world of possibilities for AI collaboration:

1. **Divide and Conquer Agent Task Execution**: Break down complex tasks into smaller, manageable tasks and let AI agents work in parallel to solve them. By leveraging the power of parallelism, large problems can be solved more quickly and efficiently.
2. **Research and Reporting Teams of Agents**: Assemble teams of AI agents to collaborate on research and reporting tasks. These teams can harness their collective intelligence to generate comprehensive and insightful analyses.
3. **Simulation-Driven Organizational Behavior Research**: Use agent trees to simulate organizational behavior and gain valuable insights. This approach can help identify patterns and areas for improvement in real-world organizations.
4. **Developer Teams of AutoGPTs Coding for You**: Create teams of AI agents that work together to develop code for your projects. By distributing coding tasks among multiple agents, complex projects can be completed more efficiently.

## Getting Started with Agent Actors

It only takes 5 minutes to learn Agent Actors. You'll need Python ^3.10, and can install it with your preferred package manager, like Poetry or Pipenv. [Check the README](https://github.com/shaman-ai/agent-actors#installation) for copy/paste instructions. Once installed, you can learn Agent Actors in just 5 minutes by diving into our [`test_system.py`]([./agent_actors/test_system.py](https://github.com/shaman-ai/agent-actors/blob/main/agent_actors/test_system.py)) file. This will help you understand how to create and manage your own agent trees.

## Contributing to Agent Actors

Agent Actors is an open-source project, and we welcome contributions from the community to help improve and expand its capabilities. Some areas where you can contribute include:

- Developing better prompts for the Plan, Do, Check, and Adjust chains
- Building visualization tools for exploring and composing execution trees
- Evaluating performance in different contexts
- Unlocking agent-to-agent communication

By exploring the Actor Model of Concurrency and harnessing the potential of parallel execution, Agent Actors provides a powerful new approach to AI collaboration. Whether you're a developer looking to build customized agent trees or a company interested in partnering to create tailored solutions, Agent Actors offers an exciting new path for AI innovation.

At Shaman AI, we're excited to partner with companies doing meaningful work to improve their workflows using agent tree architectures. We're curious to learn more about what you're working on, and how we can help &mdash; If you're interested in collaborating, please [reach out here](https://edendao.typeform.com/to/CTUCoCNy)!
