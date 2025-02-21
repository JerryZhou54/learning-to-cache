# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models.router_models import DiT_models, STE
from models.router_models import STE
from models.dynamic_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from sample import log_validation


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def format_image_to_wandb(num_router, router_size, router_scores):
    image = np.zeros((num_router, router_size, 3), dtype=np.float32)
    ones = np.ones((3), dtype=np.float32)
    for idx, score in enumerate(router_scores):
        mask = score.cpu().detach()
        for pos in range(router_size):
            image[idx, pos] = ones * mask[pos].item()
    return image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*")) - 2
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    diffusion = create_diffusion(str(args.num_sampling_steps))

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    model.load_ranking("ckpt/DDIM20_router.pt", args.num_sampling_steps, diffusion.timestep_map, 0.1)    
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    # ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    # state_dict = find_model(ckpt_path)
    # msg = model.load_state_dict(state_dict, strict=False)
    msg = model.load_state_dict(torch.load("/data/hyou37_data/learning-to-cache/DiT/results/001-DiT-XL-2/checkpoints/0104000.pt", map_location=lambda storage, loc: storage)["model"])
    if args.ratio_ckpt is not None:
        state_dict = torch.load(args.ratio_ckpt, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict["model"].items() if "diff_mod_ratio" in k}
        model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        logger.info(f"Loaded model from with msg: {msg}")
    model.eval()  # important!

    # model.add_router(args.num_sampling_steps, diffusion.timestep_map)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    #routers = [Router(len(model.module.blocks)*2) for _ in range(args.num_sampling_steps//2)]
    #routers = [DDP(r.to(device), device_ids=[rank]) for r in routers]
    opts = torch.optim.AdamW(
        # [param for name, param in model.named_parameters() if "diff_mod_ratio" not in name], 
        model.parameters(),
        lr=args.lr, weight_decay=0
    )

    # for name, param in model.named_parameters():
    #     if 'mod_router' not in name:
    #         param.requires_grad_(False)

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    from custom_dataset import CustomDataset
    features_dir = "/data/hyou37_data/imagenet_feature/imagenet256_features"
    labels_dir = "/data/hyou37_data/imagenet_feature/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    if args.wandb and rank == 0:
        import wandb
        wandb.init(
            # Set the project where this run will be logged
            project="DiT-Router", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{experiment_index:03d}-{model_string_name}", 
            # Track hyperparameters and run metadata
            config=args.__dict__
        )
        wandb.define_metric("step")
        wandb.define_metric("loss", step_metric="step")

    # Prepare models for training:
    #update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! We need to use embedding dropout for classifier-free guidance here.
    #ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_data_loss, running_l2_loss = 0, 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if train_steps < 104000:
                train_steps += 1
                continue
            x = x.to(device)
            y = y.to(device)

            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)

            #t = 1+2*torch.randint(0, diffusion.num_timesteps//2, (x.shape[0],), device=device)
            t = torch.randint(0, diffusion.num_timesteps//2, (1,), device=device)
            #t = torch.tensor(2, device=device)
            ts = t.repeat(x.shape[0])*2 + 1

            loss_dict = diffusion.training_losses(model, x, ts, model_kwargs)
            # data_loss = loss_dict["mse"].mean()
            l2_loss = loss_dict["l2_loss"].mean()
            # mod_data_loss = loss_dict["mod_mse"].mean()
            mod_data_loss = loss_dict["loss"].mean()
            # l2_loss = torch.Tensor([0]).to(mod_data_loss.device)

            #print(f"Rank: {rank}, t: {t}, data loss: {data_loss}. L1 loss: {l1_loss}")
            loss = mod_data_loss + args.l1 * l2_loss
            opts.zero_grad()
            model.zero_grad()

            loss.backward()
            #for idx, router in enumerate(model.module.routers):
            #    print(f"Rank: {rank}, idx: {idx}, ", router.prob.grad)
            opts.step()

            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         if "routers" in name:
            #             param.clamp_(-5, 5)

            # Log loss values:
            running_loss += loss.item()
            running_data_loss += mod_data_loss.item()
            running_l2_loss += l2_loss.item()

            log_steps += 1
            train_steps += 1

            model.module.reset()

            assert args.log_image_every % args.log_every == 0
            if train_steps == 1 or train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                for name, loss in [("loss", running_loss)]:
                    loss = torch.tensor(loss / log_steps, device=device)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train {name} Loss: {loss:.7f}, Train Steps/Sec: {steps_per_sec:.2f}")
            
                # scores = [model.module.routers[idx]() for idx in range(0, args.num_sampling_steps, 2)]
                
                if args.wandb and rank == 0:
                    #print(scores)
                    # mask = format_image_to_wandb(args.num_sampling_steps//2 , model.module.depth*2, scores)
                    # mask = wandb.Image(
                    #     mask,
                    # )

                    # if args.ste_threshold is not None:
                    #     final_score = [sum(STE.apply(score, args.ste_threshold)) for score in scores]
                    # else:
                    #     final_score = [sum(score) for score in scores]
                    wandb.log({
                        "step": train_steps,
                        "loss": running_loss / log_steps,
                        "mod_loss": running_data_loss / log_steps,
                        "l2_loss": running_l2_loss / log_steps
                        # "non_zero": sum(final_score)
                        # "router": mask
                    })

                if train_steps == 1 or train_steps % args.log_image_every == 0:
                    dist.barrier()
                    if args.wandb and rank == 0:
                        im = log_validation(args, model, device, vae)
                        im = wandb.Image(im)
                        wandb.log({"images": im})
                        model.train()
                    model.train()

                # Reset monitoring variables:
                running_loss = 0
                running_data_loss = 0
                running_l2_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    # router_params_list = [name for name, param in model.named_parameters() if "mod_router" in name]
                    # router_state_dict = {k: v for k, v in model.state_dict().items() if k in router_params_list}
                    checkpoint = {
                        "model": model.module.state_dict(),
                        #"ema": ema.state_dict(),
                        # "mod_routers": router_state_dict,
                        "opt": opts.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            # if train_steps > args.max_steps:
            #     print("Reach Maximum Step")
            #     break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-image-every", type=int, default=2000)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--wandb", action="store_true")
    
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ratio-ckpt", type=str, default=None)   
    #parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--l1", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=50000)

    parser.add_argument("--ste-threshold", type=float, default=None)

    args = parser.parse_args()
    main(args)
