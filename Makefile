.PHONY: install test lint format clean coverage run

install:
	poetry install

test:
	poetry run pytest -v

lint:
	poetry run ruff check .
	poetry run mypy .

format:
	poetry run black .
	poetry run isort .

clean:
	rm -f .coverage
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .venv
	rm -rf convo_filter/**/__pycache__
	rm -rf convo_filter/__pycache__
	rm -rf convo_filter/**/.pytest_cache
	rm -f filtered_convo_*.pdf.zip
	rm -f filtered_convo_*.txt.zip
	rm -f *.log

coverage:
	poetry run pytest --cov=convo_filter --cov-report=term-missing

run:
	poetry run convo-filter
