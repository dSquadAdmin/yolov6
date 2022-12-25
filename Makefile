DATASET_ROOT?=/home/ubuntu/datasets/felb
DATASET_NAME?=mix
VERSION?=yolov5x6
EPOCHS?=300
BATCH_SIZE?=16
N_GPUS?=4
DEVICES?=0,1,2,3
WEIGHTS?=''
IMG?=640
CACHE?=ram
CONF?=0.65
IOU?=0.45
DATASET_PATH=$(DATASET_ROOT)/$(DATASET_NAME)

setup:
	pip install -qr requirements.txt
# python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --device 0,1,2,3,4,5,6,7
train: setup
	python3 -m torch.distributed.run --nproc_per_node $(N_GPUS) \
	 tools/train.py --data $(DATASET_PATH)/data.yaml \
	 --epochs $(EPOCHS) \
	 --weights $(WEIGHTS) \
	 --conf configs/$(VERSION).py \
	 --img $(IMG) \
	 --batch-size $(BATCH_SIZE) \
	 --device $(DEVICES) \
	 --name $(DATASET_NAME)-$(VERSION) \
	 --cache $(CACHE)

validate:

detect:

