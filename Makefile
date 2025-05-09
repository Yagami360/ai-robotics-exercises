IMAGE_NAME=ai-robotics-exercises
IMAGE_TAG=latest

docker-build:
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

docker-run:
	docker run -it \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}:${IMAGE_TAG}

docker-exec:
	docker exec -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash
