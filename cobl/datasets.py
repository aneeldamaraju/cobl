from torch.utils.data import Dataset
import os
from natsort import natsorted
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
from .load_utils import to_cobl_path


class GroupTransforms:
    def __call__(self, sample):
        raise NotImplementedError("This method should be overridden in subclasses")


class JointRandomFlip(GroupTransforms):
    def __init__(self, horizontal_p=0.5, vertical_p=0.5):
        self.horizontal_p = horizontal_p
        self.vertical_p = vertical_p

    def __call__(self, sample):
        flip_hor = torch.rand(1)
        flip_ver = torch.rand(1)

        for idx in range(len(sample)):
            if flip_hor < self.horizontal_p:
                sample[idx] = torch.flip(sample[idx], [2])  # Assuming array is CxHxW
            if flip_ver < self.vertical_p:
                sample[idx] = torch.flip(sample[idx], [1])  # Assuming array is CxHxW

        return sample


def permute_dimensions(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = np.swapaxes(np.swapaxes(data, -3, -1), -3, -2)
    return data.astype(np.float32)


reverse_transform = transforms.Compose(
    [
        transforms.Lambda(lambda t: permute_dimensions(t)),  # CHW to HWC
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: np.clip(t, 0, 1)),
    ]
)


def CObL_collate(batch):
    return {
        "scene": torch.stack([b["scene"] for b in batch]),
        "layers": torch.stack([b["layers"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "caption": [b["caption"] for b in batch],  # list of lists
    }


class synthetic_layering_latent_v2(Dataset):
    def __init__(
        self,
        root_dir,
        dtype="float32",
        NLayer=7,
        use_aug=False,
        horizontal_fp=0.0,
        vertical_fp=0.0,
        preload=False,
    ):
        super().__init__()
        root_dir = to_cobl_path(root_dir)
        self.root_dir = root_dir
        self.files = natsorted(
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        )
        print("Found Files: ", self.files)

        if dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError("Invalid Datatype Str")

        self.NLayer = NLayer
        self.aug = JointRandomFlip(horizontal_fp, vertical_fp) if use_aug else None
        self.preload = preload
        if self.preload:
            self.data = []
            for fname in tqdm(self.files, desc="Loading Files", unit="file"):
                datpath = os.path.join(self.root_dir, fname)
                with h5py.File(datpath, "r") as dat:
                    scene = torch.tensor(np.array(dat["scene"]), dtype=self.dtype)
                    depth = torch.tensor(np.array(dat["depth"]), dtype=self.dtype)[None]
                    # We dont need the scene latent anymore
                    # scene_z = torch.tensor(np.array(dat["scene_z"]), dtype=self.dtype)
                    layers = torch.tensor(np.array(dat["layers"]), dtype=self.dtype)
                    layers = rearrange(layers, "l c h w -> (l c) h w")
                    self.data.append(
                        {
                            "scene": scene,
                            "layers": layers,
                            "caption": "",
                            "depth": depth,
                        }
                    )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.preload:
            # Return preloaded data
            dat = self.data[idx]
            if self.aug is not None:
                scene, layers, depth = self.aug(
                    [dat["scene"], dat["layers"], dat["depth"]]
                )
                dat = {
                    "scene": scene,
                    "layers": layers,
                    "caption": "",
                    "depth": depth,
                }
        else:
            # Load data on the fly
            fname = self.files[idx]
            datpath = os.path.join(self.root_dir, fname)
            with h5py.File(datpath, "r") as dat:
                scene = torch.tensor(np.array(dat["scene"]), dtype=self.dtype)
                depth = torch.tensor(np.array(dat["depth"]), dtype=self.dtype)[None]
                # scene_z = torch.tensor(np.array(dat["scene_z"]), dtype=self.dtype)
                layers = torch.tensor(np.array(dat["layers"]), dtype=self.dtype)
                layers = rearrange(layers, "l c h w -> (l c) h w")

            if self.aug is not None:
                scene, layers, depth = self.aug([scene, layers, depth])

            dat = {"scene": scene, "layers": layers, "caption": "", "depth": depth}

        return dat


if __name__ == "__main__":
    ### Test out the latent space dataset
    # rootdir = "/home/deanhazineh/ssd4tb_mounted/LDM/scripts/datasets_depth_latent/"
    # dataset = synthetic_layering_latent(root_dir=rootdir, preload=False)

    rootdir = "/home/deanhazineh/ssd4tb_mounted/LDM/scripts/full_dataset_v3/"
    dataset = synthetic_layering_latent_v2(rootdir)
    sample = dataset[100]

    scene = sample["scene"]
    layers = sample["layers"]
    caption = sample["caption"]
    depth = sample["depth"]
    print(caption)
    print(scene.shape, layers.shape, depth.shape)

    fig, ax = plt.subplots(1, 9)
    ax[0].imshow(reverse_transform(scene))
    ax[1].imshow(reverse_transform(depth))
    for i in range(5):
        ax[i + 2].imshow(reverse_transform(layers[(i * 4) : (i + 1) * 4]))
    for axi in ax.flatten():
        axi.axis("off")
    plt.show()

    # rootdir = "/home/deanhazineh/ssd4tb_mounted/LDM/scripts/datasets"
    # dataset = synthetic_layering(rootdir)
    # sample = dataset[44]
    # scene = sample["scene"]
    # layers = sample["layers"]

    # fig, ax = plt.subplots(1, 7)
    # ax[0].imshow(reverse_transform(scene))
    # for i in range(6):
    #     ax[i + 1].imshow(reverse_transform(layers[(i * 3) : (i + 1) * 3]))
    # for axi in ax.flatten():
    #     axi.axis("off")

    # plt.show()
