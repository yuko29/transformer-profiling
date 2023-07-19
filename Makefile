CONTAINER=profile
IMAGE=torch_profile
CURRENT_DIR=/home/yuko29/transformer-research
DATA_SET=/home/yysung/imagenet

startup:
	docker run -td --gpus all --ipc=host --name $(CONTAINER) \
		-v $(CURRENT_DIR):/app \
		-v $(DATA_SET):/data \
		$(IMAGE)

attach:
	docker exec -it $(CONTAINER) /bin/bash

delete:
	docker stop $(CONTAINER)
	docker rm $(CONTAINER)
