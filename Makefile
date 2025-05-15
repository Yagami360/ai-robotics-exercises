IMAGE_NAME=ai-robotics-exercises
IMAGE_TAG=latest

.PHONY: lint
lint:
	flake8 . --exclude=lerobot --exclude=openpi


.PHONY: format
format:
	isort -rc -sl --skip-glob=lerobot/** --skip-glob openpi .
	black --exclude="lerobot" --exclude="openpi" .
	isort -rc -m 3 --skip-glob=lerobot/** --skip-glob openpi .


.PHONY: docker-build
docker-build:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .


.PHONY: docker-run
docker-run:
	docker run -it \
		-v $(PWD):/app \
		-p 443:443 \
		--gpus all \
		${IMAGE_NAME}:${IMAGE_TAG}


.PHONY: docker-exec
docker-exec:
	docker exec -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash
