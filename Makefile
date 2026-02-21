.PHONY: test format

test:
	pytest -q

format:
	python -m pip install ruff
	ruff format .
