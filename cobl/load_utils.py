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
        ckpt_path = to_cobl_path(ckpt_path)
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


def replace_prefixes(state_dict, mapping):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                new_key = new_prefix + k[len(old_prefix) :]
                break  # only apply the first matching prefix
        new_state_dict[new_key] = v
    return new_state_dict


def extract_missing_checkpoint_params(ref_ckpt_path, new_ckpt_path, save_path=None):
    prefix_mapping = {
        "model.diffusion_model.": "model.",
        "cond_stage_model.model.": "text_stage_model.model.",
    }

    ref_ckpt_path = to_cobl_path(ref_ckpt_path)
    new_ckpt_path = to_cobl_path(new_ckpt_path)
    save_path = to_cobl_path(save_path)

    # Load reference checkpoint and apply prefix mapping
    ref_ckpt = torch.load(ref_ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in ref_ckpt:
        ref_state_dict = replace_prefixes(ref_ckpt["state_dict"], prefix_mapping)
    else:
        ref_state_dict = replace_prefixes(ref_ckpt, prefix_mapping)

    # Load new checkpoint
    new_ckpt = torch.load(new_ckpt_path, map_location="cpu", weights_only=False)
    new_state_dict = new_ckpt["state_dict"] if "state_dict" in new_ckpt else new_ckpt

    # Find missing keys
    ref_keys = set(ref_state_dict.keys())
    new_keys = set(new_state_dict.keys())
    missing_keys = new_keys - ref_keys

    # print(f"\nFound {len(missing_keys)} parameters in new checkpoint not in reference:")
    # for k in sorted(missing_keys):
    #     print(k)

    # Save if requested
    if save_path:
        missing_state_dict = {k: new_state_dict[k] for k in missing_keys}
        torch.save({"state_dict": missing_state_dict}, save_path)
        print(f"\nSaved missing parameters to: {save_path}")

    return missing_keys
