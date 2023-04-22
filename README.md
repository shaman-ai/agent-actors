# agi-actors: Exploring the Potential of AGI Actors through PDCA and Concurrency

Welcome to agi-actors, a proof-of-concept Python project that demonstrates the possibilities of combining AGI concepts like BabyAGI and AutoGPT with the Plan-Do-Check-Adjust (PDCA) cycle, and the actor model of concurrency for managing large language models (LLMs). agi-actors aims to inspire the LLM community, emphasizing the untapped potential of the actor model of concurrency as applied to generative agents and encouraging further exploration and development.

<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 405.12786865234375 261.60223388671875" width="1215.3836059570312" height="784.8067016601562">
  <!-- svg-source:excalidraw -->
  <defs>
    <style class="style-fonts">
      @font-face {
        font-family: "Virgil";
        src: url("https://excalidraw.com/Virgil.woff2");
      }
      @font-face {
        font-family: "Cascadia";
        src: url("https://excalidraw.com/Cascadia.woff2");
      }
    </style>
  </defs>
  <rect x="0" y="0" width="405.12786865234375" height="261.60223388671875" fill="#ffffff"></rect><g transform="translate(21.462921142578125 100.00064086914062) rotate(0 68.8099594116211 17.5)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="28px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Supervisor</text></g><g transform="translate(275.7208557128906 10) rotate(0 55.509979248046875 17.5)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="28px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Worker 1</text></g><g transform="translate(271.7599182128906 192.89703369140625) rotate(0 61.68397521972656 17.5)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="28px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Worker 2</text></g><g stroke-linecap="round"><g transform="translate(166.17229669812434 99.91926863618119) rotate(0 49.38019179938675 -20.00074566118731)"><path d="M-0.78 -0.7 C15.74 -7.8, 82.89 -34.8, 99.54 -41.54 M1.01 1.54 C17.35 -4.92, 82.51 -33.55, 98.73 -40.5" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(166.17229669812434 99.91926863618119) rotate(0 49.38019179938675 -20.00074566118731)"><path d="M75.78 -19.44 C82.53 -26.91, 90.51 -34.25, 98.22 -39.83 M77.19 -20.54 C85.2 -27.19, 90.89 -32.97, 99.64 -39.56" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(166.17229669812434 99.91926863618119) rotate(0 49.38019179938675 -20.00074566118731)"><path d="M67.61 -38.27 C77.27 -39.35, 88 -40.37, 98.22 -39.83 M69.02 -39.37 C79.49 -39.89, 87.82 -39.57, 99.64 -39.56" stroke="#000000" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(179.80511241243812 152.0867518899563) rotate(0 43.26261149779134 27.97913325763696)"><path d="M0.43 -1.06 C14.8 8.37, 70.95 47.59, 85.22 57.02 M-0.8 1 C14.01 10.07, 73.02 46.15, 87.33 55.42" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(179.80511241243812 152.0867518899563) rotate(0 43.26261149779134 27.97913325763696)"><path d="M56.23 47.23 C66.02 49.82, 69.87 53.66, 89.09 56.32 M58.43 50.06 C69.3 51.72, 78.11 53.46, 86.69 55.7" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(179.80511241243812 152.0867518899563) rotate(0 43.26261149779134 27.97913325763696)"><path d="M67.12 29.84 C74.62 36.23, 76.12 43.82, 89.09 56.32 M69.32 32.67 C76.28 40.59, 81.17 48.58, 86.69 55.7" stroke="#000000" stroke-width="1" fill="none"></path></g></g><mask></mask><g transform="translate(277.8252258300781 48.96453857421875) rotate(0 53.68797302246094 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Do and Check</text></g><g transform="translate(279.23741149902344 231.60223388671875) rotate(0 53.68797302246094 10)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="16px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Do and Check</text></g><g stroke-linecap="round"><g transform="translate(267.5461120605469 18.40331754371482) rotate(0 -93.33202151108549 35.285987398878135)"><path d="M0.95 -0.25 C-15.78 2.91, -70.08 6.98, -101.33 19.02 C-132.59 31.06, -172.43 63.19, -186.56 72 M-0.01 -1.43 C-16.83 1.94, -71.09 8.65, -102.35 20.59 C-133.62 32.54, -173.68 62.03, -187.62 70.23" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(267.5461120605469 18.40331754371482) rotate(0 -93.33202151108549 35.285987398878135)"><path d="M-169.95 46.07 C-173.44 50.86, -175.78 56.88, -186.05 68.46 M-169.92 46.57 C-173.25 52.31, -178.02 58.54, -186.73 70.72" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(267.5461120605469 18.40331754371482) rotate(0 -93.33202151108549 35.285987398878135)"><path d="M-158.79 63.29 C-164.78 64.33, -169.62 66.5, -186.05 68.46 M-158.76 63.79 C-165.05 65.07, -172.67 66.89, -186.73 70.72" stroke="#000000" stroke-width="1" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(264.5134582519531 234.69974263374252) rotate(0 -89.77789442125248 -32.77647147249874)"><path d="M-0.34 -0.86 C-16.87 -3.26, -68.29 -2.79, -98.44 -13.79 C-128.6 -24.79, -167.41 -58.33, -181.24 -66.86 M1.69 1.3 C-15.1 -1.53, -69.55 -4.52, -99.62 -15.76 C-129.7 -26.99, -165.09 -57.55, -178.78 -66.09" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(264.5134582519531 234.69974263374252) rotate(0 -89.77789442125248 -32.77647147249874)"><path d="M-151.32 -58.73 C-159.88 -62.36, -172.57 -65.66, -177.34 -66.4 M-150.81 -58.63 C-158.21 -59.32, -167.6 -62.07, -178.54 -65.43" stroke="#000000" stroke-width="1" fill="none"></path></g><g transform="translate(264.5134582519531 234.69974263374252) rotate(0 -89.77789442125248 -32.77647147249874)"><path d="M-163.29 -42.05 C-167.61 -51.68, -175.97 -61.01, -177.34 -66.4 M-162.77 -41.96 C-166.59 -47.66, -172.45 -55.31, -178.54 -65.43" stroke="#000000" stroke-width="1" fill="none"></path></g></g><mask></mask><g transform="translate(10 136.79254150390625) rotate(0 78.83995056152344 12.5)"><text x="0" y="0" font-family="Virgil, Segoe UI Emoji" font-size="20px" fill="#000000" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="text-before-edge">Plan and Review</text></g>
</svg>

## How it works

1. A **Plan** is made by a **Supervisor** for a given objective, and a semantic zero-shot topological sort is performed on the plan's tasks to adequately distribute tasks across workers.
2. **Workers** use the ReAct framework to **Do** a task and then **Check** it, improving their work if required.
3. **Supervisor** reviews the results of tasks in the **Adjust** phase and decides whether to loop back to planning or to terminate.

```
(Supervisor pid=98003) OBJECTIVE: How do we ensure the safe development of AGI?
(Supervisor pid=98003) Agent 1
(Supervisor pid=98003) [1.1] Research and implement safety measures for AGI development
(Supervisor pid=98003) [1.2] Implement explainability and interpretability features for AGI (depends on [1.1])
(Supervisor pid=98003) [1.3] Develop and implement fail-safe mechanisms for AGI (depends on [1.2])
(Supervisor pid=98003) Agent 2
(Supervisor pid=98003) [2.1] Develop and implement a comprehensive testing framework for AGI
(Supervisor pid=98003) [2.2] Conduct extensive testing on AGI using the testing framework (depends on [2.1])
(Supervisor pid=98003) [2.3] Analyze and report on the results of the testing (depends on [2.2])
(Worker pid=98046)
(Worker pid=98046) *****WORKER STARTING*****
(Worker pid=98046) [1.1] Research and implement safety measures for AGI development
(Worker pid=98047)
(Worker pid=98047) *****WORKER STARTING*****
(Worker pid=98047) [2.1] Develop and implement a comprehensive testing framework for AGI
```

## Key Features

* **Plan-Do-Check-Adjust (PDCA) Cycle**: The supervisory agent does, allowing for continuous improvement and optimization of the work distribution and execution process among worker agents.

* **Actor Model of Concurrency**: agi-actors implements the actor model, where a supervisor agent has worker agents that complete sub-tasks in parallel independently.

## Limitations

This was only tested on GPT-3.5-Turbo, and not GPT-4, because we don't have an API token.

## Request for Contributors

We invite contributors to join us in expanding agi-actors by exploring the following ideas:

1. **Improved Agent Prompts**: Develop better prompts for the Plan, Do, Check, and Adjust chains
2. **Implement Agent Memory**: Using
3. **Iterative Frontend**: Develop a frontend interface for exploring the execution tree of AGI Actors, allowing researchers to better understand and visualize the interaction between the supervisory agent and worker agents.
4. **Generalized Framework**: Build a more generalized framework for LLMs, moving beyond the specific implementation of PDCA agents and exploring other possibilities for AGI Actors.

## Acknowledgments

We extend our gratitude to the two Python packages, langchain and ray, which have significantly contributed to the development of agi-actors.

Together, let's explore the potential of AGI Actors and inspire the LLM community to delve deeper into this exciting area of research.

## Development

1. Clone the repo
2. `poetry install --with dev --with typing`

Make sure to run `poetry shell` to activate the virtual env!

### REPL-Driven Development

```bash
poetry run ipython
```

### Tests

```bash
poetry run pytest -s # run all tests
poetry run pytest -s -k 'thinking' # How can we ensure the safe development of AGI?
```
