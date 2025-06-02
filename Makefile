IMAGE_NAME=ai-robotics-exercises
IMAGE_TAG=latest
TIMESTAMP=$(shell date +%Y%m%dT%H%M)


.PHONY: lint
lint:
	flake8 . --exclude=lerobot,openpi,Isaac-GR00T,IsaacLab


.PHONY: format
format:
	isort . --skip-glob=lerobot/** --skip-glob=openpi/** --skip-glob=Isaac-GR00T/** --skip-glob=IsaacLab/**
	black --exclude="lerobot|openpi|Isaac-GR00T|IsaacLab" .
	isort -rc -m 3 --skip-glob=lerobot/** --skip-glob=openpi/** --skip-glob=Isaac-GR00T/** --skip-glob=IsaacLab/** .


.PHONY: docker-build-pi0
docker-build-pi0:
	docker build -t ${IMAGE_NAME}-pi0:${IMAGE_TAG} -f Dockerfile.pi0 .


.PHONY: docker-build-isaac
docker-build-isaac:
	docker build -t ${IMAGE_NAME}-isaac:${IMAGE_TAG} -f Dockerfile.isaac .


.PHONY: docker-run-pi0
docker-run-pi0:
	docker run -it \
		-v $(PWD):/app \
		-p 443:443 \
		--gpus all \
		${IMAGE_NAME}-pi0:${IMAGE_TAG}


.PHONY: docker-run-isaac
docker-run-isaac:
	docker run -it \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}-isaac:${IMAGE_TAG}


# .PHONY: docker-train-pi0
# docker-train-pi0:
# 	docker run -it \
# 		--rm \
# 		-v $(PWD):/app \
# 		-p 443:443 \
# 		--gpus all \
# 		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
# 		/bin/bash -c "cd /app/.debug && bash ./train.sh"


# .PHONY: docker-train-pi0-nohup
# docker-train-pi0-nohup:
# 	nohup docker run \
# 		--rm \
# 		-v $(PWD):/app \
# 		-p 443:443 \
# 		--gpus all \
# 		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
# 		/bin/bash -c "cd /app/.debug && bash ./train.sh" > train-${TIMESTAMP}.out 2>&1 &
