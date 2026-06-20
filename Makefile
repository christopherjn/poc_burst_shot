# poc-burst-shot — developer tasks
#
# `make burst` starts the FastAPI backend (the burst-shot pipeline as a service)
# and the Streamlit frontend, wires them together, and shuts the backend down
# when you quit the frontend (Ctrl-C).

BACKEND_HOST  ?= 127.0.0.1
BACKEND_PORT  ?= 8000
FRONTEND_PORT ?= 8501
BACKEND_URL    = http://$(BACKEND_HOST):$(BACKEND_PORT)
VENV          ?= .venv

.PHONY: burst backend frontend install clean

## Install core + UI dependencies into the project venv.
install:
	uv sync --group ui

## Run the FastAPI backend on its own (foreground).
backend: install
	uv run --group ui uvicorn server:app \
		--app-dir ui/backend --host $(BACKEND_HOST) --port $(BACKEND_PORT)

## Run the Streamlit frontend on its own (expects a backend already running).
frontend: install
	BURST_BACKEND_URL=$(BACKEND_URL) uv run --group ui streamlit run ui/frontend/app.py \
		--server.port $(FRONTEND_PORT) --server.address $(BACKEND_HOST)

## Start backend + frontend together and connect them.
.ONESHELL:
burst: install
	@trap 'kill $$BACKEND_PID 2>/dev/null' EXIT
	echo "Starting backend on $(BACKEND_URL) ..."
	uv run --group ui uvicorn server:app \
		--app-dir ui/backend --host $(BACKEND_HOST) --port $(BACKEND_PORT) &
	BACKEND_PID=$$!
	echo "Waiting for backend to become healthy ..."
	for i in $$(seq 1 60); do
		curl -sf $(BACKEND_URL)/health >/dev/null 2>&1 && break
		if ! kill -0 $$BACKEND_PID 2>/dev/null; then
			echo "Backend failed to start." >&2
			exit 1
		fi
		sleep 1
	done
	echo "Starting frontend on http://$(BACKEND_HOST):$(FRONTEND_PORT) ..."
	BURST_BACKEND_URL=$(BACKEND_URL) uv run --group ui streamlit run ui/frontend/app.py \
		--server.port $(FRONTEND_PORT) --server.address $(BACKEND_HOST)

clean:
	rm -rf $(VENV) **/__pycache__
