# CObL: Toward Zero-Shot Ordinal Layering without User Prompting
Official Implementation of the ICCV 2025 Paper: CObL: Toward Zero-Shot Ordinal Layering without User Prompting

## Installation
To run this code, clone and install via:
```
git clone COBL_PUBLIC_PATH
./install.sh
```
You may need to edit the torch and xformers settings in the install file to match your own CUDA version. This default file uses CUDA 12.6 and Pytorch 2.7.1


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
