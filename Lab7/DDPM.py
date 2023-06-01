import argparse
import os
import numpy as np
import math
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from UNet2D import UNet2DModel
import torch.nn as nn
import torch.nn.functional as F
import torch
from dataloader import iclevrDataset
from dataloader import getData
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from evaluator import evaluation_model
from dataclasses import dataclass

from PIL import Image
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=64,
                        help="size of each image dimension")

    parser.add_argument("--train_batch_size", type=int,
                        default=128, help="train size of the batches")

    parser.add_argument("--n_epochs", type=int, default=151,
                        help="number of epochs of training")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="gradient_accumulation_steps")

    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")

    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="warmup_steps")

    parser.add_argument('--mixed_precision', default='fp16',
                        help='`no` for float32, `fp16` for automatic mixed precision')

    parser.add_argument("--save_image_epochs", type=int,
                        default=10, help="save_image_epochs")

    parser.add_argument("--save_model_epochs", type=int,
                        default=30, help="save_model_epochs")

    parser.add_argument('--output_dir', default='ddpm-64',
                        help='the model name locally')

    parser.add_argument('--log', default='logs/',
                        help='path to tensorboard log')

    parser.add_argument("--seed", type=int,
                        default=0, help="seed")

    parser.add_argument("--overwrite_output_dir", type=bool,
                        default=True, help="overwrite the old model when re-running the notebook")

    parser.add_argument('--txt', default='accuracy.txt')

    return parser.parse_args()


evaluate_model = evaluation_model()

model = UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # the number of output channels for each UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


def evaluate(args, epoch, model, noise_scheduler, name):
    _, test_label = getData(name, "")
    x = torch.randn(len(test_label), 3, 64, 64).to(device)
    y = torch.Tensor(test_label).to(device)
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        # Get model pred
        with torch.no_grad():
            # Again, note that we pass in our labels y
            residual = model(x, t, y).sample

        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    image = (x*0.5 + 0.5).clamp(0, 1)
    # Show the results
    # fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    grid = torchvision.utils.make_grid(image, nrow=8)
    save_image(grid, f"/home/zhon/tom/dataset/images/images_{name}_{epoch}.png")
    return x, y


if __name__ == '__main__':
    args = parse_config()
    path = '/home/zhon/tom/dataset/output.txt'
    f = open(path, 'w')
    f.write("hello")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Dataloader
    train_dataloader = DataLoader(
        iclevrDataset(mode='train', root=''),
        batch_size=args.train_batch_size,
        shuffle=True
    )
    # test_dataloader = DataLoader(
    #     iclevrDataset(mode='test', root=''),
    #     batch_size=args.eval_batch_size,
    #     shuffle=False
    # )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.n_epochs),
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        # if config.push_to_hub:
        #     repo_name = get_full_repo_name(Path(config.output_dir).name)
        #     repo = Repository(config.output_dir, clone_from=repo_name)
        # elif config.output_dir is not None:
        #     os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(args.n_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (image, label) in enumerate(train_dataloader):
            clean_images = image.to(device)
            label = label.to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, label).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
                0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:

            if (epoch) % args.save_image_epochs == 0 or epoch == args.n_epochs - 1:
                name = "test"
                test_image, test_label = evaluate(args, epoch, model, noise_scheduler, name)
                acc = evaluate_model.eval(test_image, test_label)
                print(f"\nThe synthetic images of epoch {epoch} F1-score of test acc: ", acc)
                f.write(f"The synthetic images of epoch {epoch} F1-score of test acc: {acc}")

                name = "new_test"
                new_test_image, new_test_label = evaluate(
                    args, epoch, model, noise_scheduler, name)
                new_acc = evaluate_model.eval(new_test_image, new_test_label)
                print(f"\nThe synthetic images of epoch {epoch} F1-score of new test acc: ", new_acc)
                f.write(f"The synthetic images of epoch {epoch} F1-score of new test acc: {new_acc}")
    f.close()