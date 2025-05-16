IMAGE_NAME=ai-robotics-exercises
IMAGE_TAG=latest
TIMESTAMP=$(shell date +%Y%m%dT%H%M)


.PHONY: lint
lint:
	flake8 . --exclude=lerobot,openpi


.PHONY: format
format:
	isort . --skip-glob=lerobot/** --skip-glob=openpi/**
	black --exclude="lerobot|openpi" .
	isort -rc -m 3 --skip-glob=lerobot/** --skip-glob=openpi/** .


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


# .PHONY: docker-train
# docker-train:
# 	docker run -it \
# 		--rm \
# 		-v $(PWD):/app \
# 		-p 443:443 \
# 		--gpus all \
# 		${IMAGE_NAME}:${IMAGE_TAG} \
# 		/bin/bash -c "cd /app/.debug && bash ./train.sh"


# .PHONY: docker-train-nohup
# docker-train-nohup:
# 	nohup docker run \
# 		--rm \
# 		-v $(PWD):/app \
# 		-p 443:443 \
# 		--gpus all \
# 		${IMAGE_NAME}:${IMAGE_TAG} \
# 		/bin/bash -c "cd /app/.debug && bash ./train.sh" > train-${TIMESTAMP}.out 2>&1 &


.PHONY: docker-exec
docker-exec:
	docker exec -it ${IMAGE_NAME}:${IMAGE_TAG} /bin/bash
