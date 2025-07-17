from omegaconf import OmegaConf
import importlib
import torch
import os

# Update so all path should be relative to COBL root
COBL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def to_cobl_path(path):
    if path is None or path == "None":
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(COBL_ROOT, path)


def initialize_diffusion_model(
    config_path, ckpt_path=None, ignore_config_ckpts=False, override_checkpointing=False
):
    """Load the full model pipeline with a CobL checkpoint at ckpt_path."""
    # Get the config yaml and initalize the full object
    config_path = to_cobl_path(config_path)
    config = OmegaConf.load(config_path)
    config_model = config.model

    # When I am loading my own training checkpoint ckpt_path, loading the stable diffusion
    # checkpoints in the config file is redundant. That was for training initialization
    # and adds to the load time
    if ignore_config_ckpts:
        config_model.params.unet_config.ckpt_path = "None"
        config_model.params.text_stage_config.ckpt_path = "None"
        config_model.params.cond_stage_config.ckpt_path = "None"
        config_model.params.first_stage_config.ckpt_path = "None"

    if override_checkpointing:
        print("Override loaded model to use UNet Checkpointing")
        config_model.params.unet_config.params.use_checkpoint = True

    print(f"Target Module: {config_model.target}")
    diffusion_model = instantiate_from_config(config_model)

    # If given, initialize strict from a checkpoint
    # This can override all other ckpt load statements which may be empty
    # Generally use this to load a custom saved checkpoint all at once after training
    if ckpt_path is not None:
        print(f"Loading from checkpoint {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = diffusion_model.load_state_dict(sd, strict=False)

    return diffusion_model


def instantiate_from_config(config_model, ckpt_path=None, strict=False, prefix=""):
    if not "target" in config_model:
        raise KeyError("Expected key `target` to instantiate.")

    target_str = config_model["target"]
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    # Get model checkpoint (When we use SD/compvis checkpoints, we need to fix the names)
    ckpt_path = to_cobl_path(ckpt_path)
    print(ckpt_path)
    if ckpt_path is not None and ckpt_path != "None":
        print(
            f"Target: {config_model['target']} Loading from checkpoint {ckpt_path} as strict={strict} with prefix={prefix}"
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        stripped_state_dict = {
            key[len(prefix) :]: value
            for key, value in sd.items()
            if key.startswith(prefix)
        }
        missing, unexpected = loaded_module.load_state_dict(
            stripped_state_dict, strict=strict
        )
        print(
            f"Restored {target_str} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return loaded_module


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)
