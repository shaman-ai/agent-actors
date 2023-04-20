# PDCA-GPT

## TODO

1. Get the test cases to work reasonably!
2. Consider refactoring the structure of `agent.py` into a pure plan-do-check-adjust

## Development

1. Clone the repo
2. `poetry install --with dev --with typing`

Make sure to run `poetry shell` to activate the virtual env!

### REPL-Driven Development

```bash
poetry run ipython
```

Then in IPython run:

```python
from importlib import reload
```

### Tests

```bash
poetry run pytest -s # run all tests
poetry run pytest -s -k 'logic' # only run tests that include 'logic' in the function name
```

