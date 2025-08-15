import os
import zipfile
import requests
from tqdm import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from cobl.load_utils import to_cobl_path

# REAL_PATH = "https://www.dropbox.com/scl/fi/ecs1nco5xep9gqk4k0mvr/tabletop_test.zip?rlkey=uo83pdgg8063qle6ljoyaym12&st=7fzwd1vs&dl=1"
REAL_PATH = "https://www.dropbox.com/scl/fi/4r8mm3cvtik54sg9c75f5/test.zip?rlkey=lut3xa69rv8rxizoz0qdnl52y&st=3o4b4bz9&dl=1"
SYN_PATH = "https://www.dropbox.com/scl/fi/2kapzajj9srzgissnw2jt/train.zip?rlkey=r5uqz0o6g5oodpi5r43hvqgz1&st=b55blbi7&dl=1"


class Tabletop(Dataset):
    def __init__(self, split="test"):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.split = split

        # build a cache path, e.g. ~/.cobl/TableTop_Cache/test/
        cache_root = to_cobl_path("cobl/TableTop_Cache")
        os.makedirs(cache_root, exist_ok=True)
        split_dir = os.path.join(cache_root, split)
        os.makedirs(split_dir, exist_ok=True)

        # if no .h5 files yet, download & extract
        h5_files = [f for f in os.listdir(split_dir) if f.endswith(".h5")]

        if not h5_files:
            print(f"[Tabletop] Downloading {split} split to cache…")
            zip_path = os.path.join(cache_root, f"{split}.zip")
            self._download_with_progress(
                REAL_PATH if split == "test" else SYN_PATH, zip_path
            )
            print(zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(cache_root)
            os.remove(zip_path)
            print(f"[Tabletop] Extracted to {cache_root}")

        self.files = sorted(
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".h5")
        )
        if not self.files:
            raise RuntimeError(f"No .h5 files found in {split_dir}")

    def _download_with_progress(self, url: str, dest: str, chunk_size=1024):
        """Stream‑download `url` to `dest` with a tqdm progress bar."""
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as fp, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading to {dest}"
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fp.write(chunk)
                pbar.update(len(chunk))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        keys = (
            ["background", "scene", "masks", "layers", "shadows", "descriptions"]
            if self.split == "train"
            else ["background", "scene", "masks", "layers"]
        )

        dat = {"filename": os.path.basename(path)}
        with h5py.File(path, "r") as f:
            for k in keys:
                arr = f[k][()]
                if arr.dtype != np.uint8 and np.issubdtype(arr.dtype, np.number):
                    arr = arr.astype(np.uint8)
                dat[k] = arr

        return dat


if __name__ == "__main__":
    test = Tabletop(split="test")
    dat = test[0]
    scene = dat["scene"]
    background = dat["background"]
    mask = dat["masks"]
    layers = dat["layers"]
    filename = dat["filename"]
    print(filename, scene.shape, mask.shape, layers.shape, background.shape)

    fig, ax = plt.subplots(1, 6)
    for i in range(6):
        ax[i].imshow(layers[i])

    train = Tabletop(split="train")
    dat = train[0]

    scene = dat["scene"]
    background = dat["background"]
    masks = dat["masks"]
    layers = dat["layers"]
    shadows = dat["shadows"]
    filename = dat["filename"]
    descriptions = dat["descriptions"]
    print(
        filename,
        descriptions,
        scene.shape,
        masks.shape,
        layers.shape,
        background.shape,
        shadows.shape,
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(scene)
    ax[0].set_title(descriptions)
    ax[1].imshow(background)
    for axi in ax.flatten():
        axi.axis("off")

    fig, ax = plt.subplots(3, 6)
    for i in range(6):
        ax[0, i].imshow(layers[i])
        ax[1, i].imshow(masks[i])
        ax[2, i].imshow(shadows[i])
    for axi in ax.flatten():
        axi.axis("off")

    plt.show()
