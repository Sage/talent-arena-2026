all: libs test
 
libs:
	uv pip compile --all-extras pyproject.toml --output-file requirements.txt

test:
	uv run pytest -s test_tools.py