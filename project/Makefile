IMAGE_NAME        = audio-pipeline-base
TAG               = latest
DOCKER_BUILDKIT   ?= 0
SCRIPT            = gradio_pipeline.py
VENV_DIR          = venv
SHELL := /bin/bash

.PHONY: build build-base up down clean run stop run-script-local run-script-docker

build-base:
	@export $$(grep HUGGINGFACEHUB_API_TOKEN .env) && \
	docker build \
		--build-arg HUGGINGFACEHUB_API_TOKEN=$$HUGGINGFACEHUB_API_TOKEN \
		-t $(IMAGE_NAME):$(TAG) \
		-f Dockerfile.base .

build:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker compose -build

up:
	docker compose up -d

down:
	docker compose down

clean:
	docker system prune -f

run: build up

stop: down clean

run-script-docker: build-base
	docker run --rm \
		-v "$(shell pwd)":/app \
		-w /app \
		$(IMAGE_NAME):$(TAG) \
		python3 $(SCRIPT)