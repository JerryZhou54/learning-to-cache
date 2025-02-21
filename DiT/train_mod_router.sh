CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 --master_port 12345 train_router.py \
	--model DiT-XL/2 \
	--data-path /data/hyou37_data/imagenet_feature \
	--global-batch-size 32 \
	--image-size 256 \
	--ckpt-every 4000 \
	--l1 1 \
	--lr 2e-5 \
	--wandb \
	--ste-threshold 0.1 \
	--epochs 5