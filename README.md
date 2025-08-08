# CObL: Toward Zero-Shot Ordinal Layering without User Prompting (ICCV 2025 Highlight)

Helpful links: [\[Project Page\]](https://vision.seas.harvard.edu/cobl/) [\[PDF\]]() [\[ArXiv\]]()

Teaser

## Installation
To run this code, clone and install via:
```
git clone COBL_PUBLIC_PATH
./install.sh
```
You may need to edit the torch and xformers settings in the install file to match your own CUDA version. This default file uses CUDA 12.6 and Pytorch 2.7.1, Python 3.10.13


## TableTop Real - 20 testing Subset
You can access the Tabletop testing set (real-world scenes) by:

(1) Cloning and installing this github repository, then calling:
```
from cobl.Tabletop import Tabletop
test_dat = Tabletop(split="test")
```
(2) or without downloading this CoBL Repository, by loading from Huggingface:
```
from datasets import load_dataset
cobl = load_dataset("DeanHazineh1/CoBL_Tabletop")
sample = cobl['test'][0]
```

## TableTop Synthetic - Training Subset


## Citation

If you find this repo useful, please consider citing:

```bibtex
@article{damaraju2025cobl,
      author    = {Damaraju, Aneel and Hazineh, Dean and Zickler, Todd},
      title     = {CObL: Toward Zero-Shot Ordinal Layering without User Prompting},
      journal   = {ICCV},
      year      = {2025},
    }
```
