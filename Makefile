.PHONY: install lint test data eval-judge eval-judge-dev demo-gate serve up down

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

data:            ## download public datasets into data/raw and build the splits
	./scripts/download_data.sh
	uv run python evals/mtbench/prepare.py
	uv run python examples/banking77/prepare.py

eval-judge-dev:  ## judge calibration on the MT-Bench dev split (iterate here)
	uv run python evals/mtbench/run.py --split dev

eval-judge:      ## judge calibration on the MT-Bench test split (the reported numbers)
	uv run python evals/mtbench/run.py --split test

demo-gate:       ## re-gate the committed Banking77 runs (no API key needed)
	-uv run assaygate compare examples/banking77/runs/baseline-v1 examples/banking77/runs/rerun-v1
	-uv run assaygate compare examples/banking77/runs/baseline-v1 examples/banking77/runs/candidate-v2 \
		--html examples/banking77/runs/gate-report.html

serve:
	uv run assaygate serve --runs-dir examples/banking77/runs

up:
	docker compose up --build -d

down:
	docker compose down
