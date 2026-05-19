.PHONY: help install test lint type api demo docker clean

PYTHON := python3.11
PKG    := ongomemory

help:
	@echo "ongomemory — living memory layer for Ongo"
	@echo ""
	@echo "  install     install package + dev deps in editable mode"
	@echo "  test        run pytest with coverage"
	@echo "  lint        ruff check + format"
	@echo "  type        mypy strict"
	@echo "  api         launch fastapi server (port 8002)"
	@echo "  demo        run the 3-day scripted demo through the CLI"
	@echo "  docker      build edge image"
	@echo "  clean       remove build / cache artifacts"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=$(PKG) --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

type:
	$(PYTHON) -m mypy src/$(PKG)

api:
	$(PYTHON) -m uvicorn $(PKG).api:app --host 0.0.0.0 --port 8002 --reload

demo:
	$(PYTHON) -m $(PKG).cli demo --user f7a2

docker:
	docker buildx build -t ongomemory/edge:latest -f docker/Dockerfile .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
