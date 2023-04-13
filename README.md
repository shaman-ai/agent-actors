# PDCA-GPT

## Development

1. Clone the repo
2. `poetry install --with dev --with typing`

Make sure to run `poetry shell` to activate the virtual env!

### Tests

```bash
poetry run pytest -s # run all tests
poetry run pytest -s -k 'math' # only run tests that include 'math' in the function name
```

