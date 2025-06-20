IMAGE_NAME = audio-pipeline-base

TAG = latest

DOCKER_BUILDKIT ?= 0

build:
	DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker compose build $(IMAGE_NAME)

up:
	docker compose up -d

down:
	docker compose down

clean:
	docker system prune -f

run: build up

stop: down clean
