import os
import zipfile
import requests
from tqdm import tqdm
import h5py
import numpy as np
from torch.utils.data import Dataset
from cobl.load_utils import to_cobl_path

REAL_PATH = "https://www.dropbox.com/scl/fi/ecs1nco5xep9gqk4k0mvr/tabletop_test.zip?rlkey=uo83pdgg8063qle6ljoyaym12&st=7fzwd1vs&dl=1"


class Tabletop(Dataset):
    def __init__(self, split="test"):
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.split = split

        # build a cache path, e.g. ~/.cobl/TableTop_Cache/test/
        cache_root = to_cobl_path("cobl/TableTop_Cache")
        split_dir = os.path.join(cache_root, split)
        os.makedirs(split_dir, exist_ok=True)

        # if no .h5 files yet, download & extract
        h5_files = [f for f in os.listdir(split_dir) if f.endswith(".h5")]
        if not h5_files:
            print(f"[Tabletop] Downloading {split} split to cache…")
            zip_path = os.path.join(cache_root, f"{split}.zip")
            self._download_with_progress(REAL_PATH, zip_path)
            print(zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(split_dir)
            os.remove(zip_path)
            print(f"[Tabletop] Extracted to {split_dir}")

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
        with h5py.File(path, "r") as f:
            scene = f["scene"][:].astype(np.uint8)
            masks = f["masks"][:].astype(np.uint8)
            layers = f.get("layers")
            composites = f.get("composites")
            layers = layers[:].astype(np.uint8) if layers is not None else None
            composites = (
                composites[:].astype(np.uint8) if composites is not None else None
            )

        return {
            "scene": scene,  # (H, W, 3)
            "masks": masks,  # (N, H, W)
            "layers": layers,  # (N, H, W, 3) or None
            "composites": composites,  # (N, H, W, 3) or None
            "filename": os.path.basename(path),
        }


if __name__ == "__main__":
    test = Tabletop()
    dat = test[0]
    scene = dat["scene"]
    mask = dat["masks"]
    layers = dat["layers"]
    composites = dat["composites"]
    filename = dat["filename"]
    print(filename, scene.shape, mask.shape, layers.shape, composites.shape)
