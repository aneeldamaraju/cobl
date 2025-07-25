import torch
import numpy as np
import torch.nn as nn


class MidasDepthEstimator(nn.Module):
    def __init__(self, model_type="DPT_Large"):
        super().__init__()
        """
        Initializes the MiDaS depth estimator model and the required transforms.
        """
        # Load the MiDaS model
        self.model_type = model_type
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.eval()

        # Load MiDaS transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def forward(self, img):
        """
        Given an input image (NumPy array in BGR or RGB format),
        perform inference and return the depth map as a NumPy array.
        """
        # Prepare input
        device = next(self.midas.parameters()).device
        input_batch = self.transform(img).to(device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction[None] / prediction.max()

    def midas_prep(self, x):
        """Convert a tensor input to match the default midas transform"""
        x = x[:3, ...].permute(1, 2, 0).detach().cpu().numpy() * 255
        return x.astype(np.uint8)
