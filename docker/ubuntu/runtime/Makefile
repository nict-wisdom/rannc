CUDA_VERSION = 11.3
OS_NAME = ubuntu
OS_VERSION = 16.04
PYTHON_VERSION = 3.7

IMG_NAME = rannc_runtime_cuda$(CUDA_VERSION)-$(OS_NAME)$(OS_VERSION)-py$(PYTHON_VERSION)

BASE_IMG_ID = base_img_id
RUNTIME_IMG_ID = runtime_img_id
CWD = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD_DIR = $(abspath $(CWD)/../../..)
WHEEL_PATH_FILE = ../package/wheel_path
WHEEL_FULL_PATH = $(shell cat $(WHEEL_PATH_FILE))
WHEEL_FILE = $(shell basename $(WHEEL_FULL_PATH))
OUTPUT_DIR = $(abspath $(CWD)/../outputs)
DOCKER_IMAGE_FILE = $(OUTPUT_DIR)/$(IMG_NAME).tar
SING_IMAGE_FILE = $(OUTPUT_DIR)/$(IMG_NAME).sif

all : $(SING_IMAGE_FILE)

$(SING_IMAGE_FILE): runtime_img_id
ifeq (,$(wildcard $(SING_IMAGE_FILE)))
	@echo "Singulariy image file exists. Overwriting ..."
endif
	mkdir -p $(OUTPUT_DIR)
	@echo "Exporting container to $(OUTPUT_DIR)/$(IMG_NAME).tar ..."
	docker save $(IMG_NAME) > $(DOCKER_IMAGE_FILE)
	@echo "Converting container $(DOCKER_IMAGE_FILE) to $(SING_IMAGE_FILE) ..."
	sed -e "s/___RUNTIME_IMAGE___/$(IMG_NAME)/" singularity.def.template > singularity.def
	singularity build --force --fakeroot $(SING_IMAGE_FILE) singularity.def
	rm -f $(DOCKER_IMAGE_FILE)

runtime_img_id: ../base/$(BASE_IMG_ID) Dockerfile $(WHEEL_PATH_FILE)
	cp $(WHEEL_FULL_PATH) .

	docker build -t $(IMG_NAME) \
		--build-arg PKG_FILE=./$(WHEEL_FILE) \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg OS_NAME=$(OS_NAME) \
		--build-arg OS_VERSION=$(OS_VERSION) \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		.
	docker inspect -f "{{ .ID }}" $(IMG_NAME) > ./$(RUNTIME_IMG_ID)
	rm ./$(WHEEL_FILE)

clean:
ifeq (,$(wildcard ./$(RUNTIME_IMG_ID)))
	@echo "Nothing to clean."
else
	docker rmi $(shell cat ./$(RUNTIME_IMG_ID))
	rm ./$(RUNTIME_IMG_ID)
endif
	rm -f $(DOCKER_IMAGE_FILE)
	rm -f $(SING_IMAGE_FILE)
