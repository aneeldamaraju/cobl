from omegaconf import OmegaConf
import importlib
import torch
from torch.utils.data import DataLoader, random_split
import os
import itertools  # NEVER USE ITERTOOLS.CYCLE ON TRAINING DATA WITH RANDOM AUGMENTATIONS
from torch.optim import AdamW
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from cobl import reverse_transform, DDIM_Sampler
from .load_utils import *
from .datasets import CObL_collate
from .load_utils import to_cobl_path


def initialize_training(
    config_path,
    model_ckpt_path=None,
    override_eager=False,
):
    config_path = to_cobl_path(config_path)
    config = OmegaConf.load(config_path)

    config_trainer = config.trainer
    config_train_dat = config.train_dataset
    config_valid_dat = (
        config.valid_dataset if "valid_dataset" in config.keys() else None
    )

    ## used for quick testing:
    if override_eager:
        config_train_dat.params.preload = False

    # Instantiate the model
    model = initialize_diffusion_model(config_path, model_ckpt_path)
    model = model.to("cuda")

    # Load the datasets
    train_dataset = instantiate_from_config(config_train_dat)
    valid_dataset = (
        instantiate_from_config(config_valid_dat)
        if config_valid_dat is not None
        else None
    )

    # Initialize the trainer
    print(f"Trainer Module: {config_trainer.target}")
    trainer_params = config_trainer.get("params", dict())
    trainer = get_obj_from_str(config_trainer["target"])(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        **trainer_params,
    )

    return trainer


def step_scheduler(scheduler, epoch, original_T_max):
    if epoch >= original_T_max:
        for param_group in scheduler.optimizer.param_groups:
            param_group["lr"] = scheduler.eta_min
    else:
        scheduler.step(epoch)
    return


# Compute the Exponential Moving Average (EMA)
def compute_ema(data, alpha=0.1):
    ema = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return np.array(ema)


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        ckpt_path,
        batch_size,
        max_steps,  # In this version this maps to epochs
        lr,
        gradient_accumulation_steps,
        snapshot_every_n,
        sample_img_size,
        disp_num_samples,
        load_optimizer=True,
        valid_dataset=None,
        train_valid_split=0.85,
        dl_workers=1,
        dl_pin_mem=True,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        exp_decay_lr=1e-6,
        alpha=0.9999,
        hard_reset_lr=False,
        preencoded=False,
    ):
        self.model = model.to("cuda")
        self.target_key = model.target_key
        self.text_key = model.text_key
        self.cond_key = model.cond_key

        if valid_dataset is None:
            torch.manual_seed(42)
            train_size = int(train_valid_split * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(
                train_dataset, [train_size, valid_size]
            )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dl_workers,
            pin_memory=dl_pin_mem,
            collate_fn=CObL_collate,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=dl_pin_mem,
            collate_fn=CObL_collate,
        )

        ckpt_path = to_cobl_path(ckpt_path)
        self.ckpt_path = ckpt_path
        self.max_epochs = max_steps
        self.lr = lr
        self.gaccum_steps = gradient_accumulation_steps
        self.snapshot_every_n = snapshot_every_n
        self.sample_img_size = sample_img_size
        self.snum = batch_size if disp_num_samples > batch_size else disp_num_samples
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.load_optimizer = load_optimizer
        self.exp_decay_lr = exp_decay_lr
        self.alpha = alpha
        self.hard_reset_lr = hard_reset_lr
        self.preencoded = preencoded

        for fd in [
            ckpt_path,
            os.path.join(ckpt_path, "model_snapshots/"),
            os.path.join(ckpt_path, "train_logs/"),
        ]:
            os.makedirs(fd, exist_ok=True)

        ### In inherited class you can change model parameters
        model_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                model_params.append(param)
        self.model_params = model_params

    def fit(self):
        model = self.model
        train_dl = self.train_dataloader
        valid_dl = self.valid_dataloader
        valid_iter = itertools.cycle(valid_dl)
        train_iter = itertools.cycle(train_dl)  # Used only for visualization data

        ckpt_folder = self.ckpt_path + "model_snapshots/"
        log_folder = self.ckpt_path + "train_logs/"
        last_ckpt_path = ckpt_folder + "ckpt_last.ckpt"
        best_ckpt_path = ckpt_folder + "best_ckpt.ckpt"

        # Fix grad accumulation step number as precaution
        dl_len = len(train_dl)
        gaccum_steps = dl_len if self.gaccum_steps > dl_len else self.gaccum_steps
        print(f"Gradient accumulation steps: {gaccum_steps}")

        # Set up optimizer
        print("Learning Rate: ", self.lr)
        optimizer = AdamW(
            self.model_params,
            self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.exp_decay_lr,
        )

        # Load the last checkpoint to resume paused training
        if os.path.exists(last_ckpt_path):
            checkpoint = torch.load(last_ckpt_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            model.to("cuda")

            optimizer_ckpt = checkpoint["optimizer_state_dict"]
            start_epoch = checkpoint["start_epoch"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            epoch_vec = checkpoint["epoch_vec"]

            if self.load_optimizer:
                optimizer.load_state_dict(optimizer_ckpt)
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                alpha = checkpoint["alpha"]
                ema_grads = checkpoint["ema_grads"]
                ema_grads = {
                    key: tensor.to("cuda") for key, tensor in ema_grads.items()
                }
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_epochs,
                    eta_min=self.exp_decay_lr,
                )
                alpha = self.alpha
                ema_grads = {
                    name: torch.zeros_like(param).to("cuda")
                    for name, param in model.named_parameters()
                }
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            model.to("cuda")
            start_epoch = 0
            train_losses = []
            test_losses = []
            train_metrics = []
            test_metrics = []
            epoch_vec = []
            ema_grads = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
            }
            alpha = self.alpha
        original_T_max = scheduler.state_dict()["T_max"]
        print("original t max: ", original_T_max)

        if os.path.exists(best_ckpt_path):
            best_state = torch.load(best_ckpt_path, map_location="cpu")
        else:
            best_state = {}

        # Run Training
        for epoch in range(start_epoch, self.max_epochs + 1):
            start_time = time.time()
            model.train()
            epoch_loss = 0

            for step, batch in enumerate(train_dl):
                start = time.time()
                optimizer.zero_grad()
                loss = model.training_step(batch, self.preencoded)
                loss.backward()
                end = time.time()
                print(
                    f"Epoch Progress: {step / len(train_dl)} Step {step} Loss {loss.item()} Elapsed Time: {end-start}"
                )

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        ema_grads[name] = (
                            alpha * param.grad + (1 - alpha) * ema_grads[name]
                        )
                        param.grad = ema_grads[name]

                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_dl)
            train_losses.append(epoch_loss)
            if self.hard_reset_lr:
                scheduler.step()
            else:
                step_scheduler(scheduler, epoch, original_T_max)

            # Valid epoch
            model.eval()
            epoch_loss = 0
            for batch in valid_dl:
                with torch.no_grad():
                    loss = model.training_step(batch, self.preencoded)
                    epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(valid_dl)
            test_losses.append(epoch_loss)

            end_time = time.time()
            current_lr = [optimizer.param_groups[0]["lr"]]
            time_elapsed = end_time - start_time
            print(
                f"Epoch {epoch+1} Finished in {time_elapsed:.2f}| lr: {current_lr} Train_loss: {train_losses[-1]:.2e} Test_Loss: {test_losses[-1]:.2e}"
            )

            prev_min = np.min(test_losses[:-1] + [1e6])
            if test_losses[-1] <= prev_min:
                print(f"New Best Loss {test_losses[-1]} vs {prev_min}")
                best_state = {
                    "start_epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "epoch_vec": epoch_vec,
                    "ema_grads": ema_grads,
                    "alpha": alpha,
                }

            if epoch % self.snapshot_every_n == 0 or epoch == self.max_epochs:
                # Save a loss figure
                ema_train = compute_ema(train_losses, alpha=0.10)
                ema_test = compute_ema(test_losses, alpha=0.10)
                plt.figure(figsize=(10, 5))
                ax = plt.gca()
                ax.plot(train_losses, ".", color="gray", label="train", alpha=0.1)
                ax.plot(ema_train, "r-", label="train", linewidth=2, alpha=0.7)
                ax.plot(ema_test, "g-", label="valid", linewidth=2, alpha=0.7)
                ax.set_xlabel("batch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                lines1, labels1 = ax.get_legend_handles_labels()
                ax.legend(lines1, labels1, loc=0)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.tight_layout()
                plt.savefig(log_folder + "training_loss.png")
                plt.close()

                # Draw some samples for visualization
                self.plot_drawn_samples(next(train_iter), f"epoch_{epoch}_train.png")
                self.plot_drawn_samples(next(valid_iter), f"epoch_{epoch}_test.png")
                epoch_vec.append(epoch)

                state = {
                    "start_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "epoch_vec": epoch_vec,
                    "ema_grads": ema_grads,
                    "alpha": alpha,
                }
                torch.save(state, last_ckpt_path)
                print("Saved Model Checkpoint!")

                if best_state:
                    torch.save(best_state, best_ckpt_path)

                time.sleep(1)

        return

    def plot_drawn_samples(self, batch, fname):
        return None


class Trainer_with_plotting(Trainer):
    def __init__(self, ckpt_to_find_missing_params, unfreeze_unet=False, **kwargs):
        super().__init__(**kwargs)

        # Limit the training parameters
        prefix_correspond = {
            "model.": "model.diffusion_model.",
            "first-stage_model.": "first_stage_model.",
            "text_stage_model.": "cond_stage_model.",
        }

        # Load checkpoint
        ckpt_path = to_cobl_path(ckpt_to_find_missing_params)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        mapped_ckpt_keys = set()
        for ckpt_key in sd.keys():
            for model_prefix, ckpt_prefix in prefix_correspond.items():
                if ckpt_key.startswith(ckpt_prefix):
                    mapped_key = model_prefix + ckpt_key[len(ckpt_prefix) :]
                    mapped_ckpt_keys.add(mapped_key)

        model_keys = set(name for name, _ in self.model.named_parameters())
        missing_keys = model_keys - mapped_ckpt_keys
        trainable_missing_keys = {
            name
            for name, param in self.model.named_parameters()
            if name in missing_keys and param.requires_grad
        }

        if unfreeze_unet:
            unfreeze_keys = {
                name
                for name, param in self.model.named_parameters()
                if name.startswith("model.") and param.requires_grad
            }
            trainable_missing_keys |= unfreeze_keys  # Union them in

        model_params = []
        for name, param in self.model.named_parameters():
            if name in trainable_missing_keys:
                param.requires_grad = True
                print(name)
                model_params.append(param)
        self.model_params = model_params

    def plot_drawn_samples(self, batch, fname):
        sampler = DDIM_Sampler(self.model, 50)

        text_cond = batch[self.text_key]
        cond = batch[self.cond_key]
        target = batch[self.target_key]
        if "depth" in batch.keys():
            depth = batch["depth"]
        else:
            depth = None

        snum = np.minimum(self.snum, target.shape[0])
        text_cond = text_cond[:snum]
        cond = cond[:snum]
        target = target[:snum]

        ximgs = sampler.sample(
            image_size=[64, 64],
            batch_size=snum,
            text_cond=text_cond,
            cond=cond,
            depth=depth,
        )
        print(ximgs.shape)

        if snum > 1:
            fig, ax = plt.subplots(snum, 8, figsize=(8 * 3, snum * 3))
            for i in range(snum):
                ax[i, 0].imshow(reverse_transform(cond[i]))
                for j in range(7):
                    ax[i, j + 1].imshow(
                        reverse_transform(ximgs[0, i, j * 3 : (j + 1) * 3])
                    )
        else:
            fig, ax = plt.subplots(1, 8, figsize=(8 * 3, 3))
            ax[0].imshow(reverse_transform(cond[0]))
            for j in range(7):
                ax[j + 1].imshow(reverse_transform(ximgs[0, 0, j * 3 : (j + 1) * 3]))

        for axi in ax.flatten():
            axi.axis("off")

        plt.tight_layout()
        plt.savefig(self.ckpt_path + f"train_logs/{fname}", bbox_inches="tight")
        plt.close()
        return
