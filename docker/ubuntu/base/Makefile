CUDA_VERSION = 11.3
OS_NAME = ubuntu
OS_VERSION = 18.04
PYTHON_VERSION = 3.8.14

IMG_NAME = rannc_base_cuda$(CUDA_VERSION)-$(OS_NAME)$(OS_VERSION)-py$(PYTHON_VERSION)
BASE_IMG_ID = base_img_id

.PHONY: clean

all : base

base: Dockerfile
	docker build -t $(IMG_NAME) \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg OS_NAME=$(OS_NAME) \
		--build-arg OS_VERSION=$(OS_VERSION) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		.
	docker inspect -f "{{ .ID }}" $(IMG_NAME) > ./$(BASE_IMG_ID)

clean:
ifeq (,$(wildcard ./$(BASE_IMG_ID)))
	@echo "Nothing to clean."
else
	docker rmi $(shell cat ./$(BASE_IMG_ID))
	rm ./$(BASE_IMG_ID)
endif
