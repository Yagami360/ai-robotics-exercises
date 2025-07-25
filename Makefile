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


.PHONY: docker-build-genesis
docker-build-genesis:
	cd Genesis && docker build -t ${IMAGE_NAME}-genesis:${IMAGE_TAG} -f docker/Dockerfile docker


# .PHONY: docker-build-cosmos-predict2
# docker-build-cosmos-predict2:
# 	cd cosmos-predict2 && docker build -t ${IMAGE_NAME}-cosmos-predict2:${IMAGE_TAG} -f Dockerfile .


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


.PHONY: docker-run-genesis
docker-run-genesis:
	xhost +local:root
	docker run --gpus all --rm -it \
		-e DISPLAY=:1 \
		-v /dev/dri:/dev/dri \
		-v /tmp/.X11-unix/:/tmp/.X11-unix \
		-v $(PWD):/workspace \
		${IMAGE_NAME}-genesis:${IMAGE_TAG}


.PHONY: docker-run-cosmos-predict2
docker-run-cosmos-predict2:
	docker run -it \
		-v $(PWD)/cosmos-predict2:/workspace \
		-v $(PWD)/cosmos-predict2/datasets:/workspace/datasets \
		-v $(PWD)/cosmos-predict2/checkpoints:/workspace/checkpoints \
		--gpus all \
		nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.1


.PHONY: docker-run-cosmos-transfer1
docker-run-cosmos-transfer1:
	docker run -it \
		-v $(PWD)/cosmos-transfer1:/workspace \
		-v $(PWD)/cosmos-transfer1/checkpoints:/workspace/checkpoints \
		--gpus all \
		nvcr.io/${USER}/cosmos-transfer1:latest


.PHONY: docker-train-pi0-aloha
docker-train-pi0-aloha:
	docker run -it \
		--rm \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
		/bin/bash -c "cd /app/.debug && bash ./train_pi0_aloha.sh"


.PHONY: docker-train-pi0-aloha-nohup
docker-train-pi0-aloha-nohup:
	nohup docker run \
		--rm \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
		/bin/bash -c "cd /app/.debug && bash ./train_pi0_aloha.sh" > train-pi0-aloha-${TIMESTAMP}.out 2>&1 &


.PHONY: docker-train-act-aloha
docker-train-act-aloha:
	docker run -it \
		--rm \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
		/bin/bash -c "cd /app/.debug && bash ./train_act_aloha.sh"


.PHONY: docker-train-act-aloha-nohup
docker-train-act-aloha-nohup:
	nohup docker run \
		--rm \
		-v $(PWD):/app \
		--gpus all \
		${IMAGE_NAME}-pi0:${IMAGE_TAG} \
		/bin/bash -c "cd /app/.debug && bash ./train_act_aloha.sh" > train-act-aloha-${TIMESTAMP}.out 2>&1 &
