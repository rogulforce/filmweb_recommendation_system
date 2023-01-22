poetry run isort $1
poetry run black $1
poetry run mypy $1
# poetry run flake8 $1
poetry run pylint $1