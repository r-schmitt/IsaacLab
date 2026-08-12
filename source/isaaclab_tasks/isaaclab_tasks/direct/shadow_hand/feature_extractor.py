# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os

import torch
import torch.nn as nn
import torchvision

from isaaclab.sensors import save_images_to_file
from isaaclab.utils import configclass


def _conv_output_size(size: int, kernel: int, stride: int) -> int:
    """Spatial size after a ``Conv2d``/pool with the given kernel and stride (no padding)."""
    return (size - kernel) // stride + 1


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self, num_channels: int = 3, height: int = 64, width: int = 64):
        super().__init__()
        # The LayerNorm normalized shapes depend on the input resolution, so they are
        # derived from the conv stack rather than hard-coded — the net stays valid if the
        # camera resolution changes.
        conv_params = ((6, 2), (4, 2), (4, 2), (3, 2))
        conv_channels = (16, 32, 64, 128)
        h, w = height, width
        spatial: list[list[int]] = []
        for kernel, stride in conv_params:
            h = _conv_output_size(h, kernel, stride)
            w = _conv_output_size(w, kernel, stride)
            spatial.append([h, w])

        in_channels = (num_channels,) + conv_channels[:-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels[0], conv_channels[0], kernel_size=conv_params[0][0], stride=conv_params[0][1]),
            nn.ReLU(),
            nn.LayerNorm([conv_channels[0], *spatial[0]]),
            nn.Conv2d(in_channels[1], conv_channels[1], kernel_size=conv_params[1][0], stride=conv_params[1][1]),
            nn.ReLU(),
            nn.LayerNorm([conv_channels[1], *spatial[1]]),
            nn.Conv2d(in_channels[2], conv_channels[2], kernel_size=conv_params[2][0], stride=conv_params[2][1]),
            nn.ReLU(),
            nn.LayerNorm([conv_channels[2], *spatial[2]]),
            nn.Conv2d(in_channels[3], conv_channels[3], kernel_size=conv_params[3][0], stride=conv_params[3][1]),
            nn.ReLU(),
            nn.LayerNorm([conv_channels[3], *spatial[3]]),
            nn.AvgPool2d(spatial[3]),
        )

        self._feature_dim = conv_channels[-1]
        self.linear = nn.Sequential(
            nn.Linear(self._feature_dim, 27),
        )

        self.data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x[:, 0:3, :, :] = self.data_transforms(x[:, 0:3, :, :])
        cnn_x = self.cnn(x)
        out = self.linear(cnn_x.view(-1, self._feature_dim))
        return out


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, log_dir: str | None = None, height: int = 64, width: int = 64):
        """Initialize the feature extractor model.

        Args:
            cfg: Configuration for the feature extractor model.
            device: Device to run the model on.
            log_dir: Directory to save checkpoints. Default is None, which uses the local
                "logs" folder resolved relative to this file.
            height: Height of the input camera image, used to size the CNN. Default is 64.
            width: Width of the input camera image, used to size the CNN. Default is 64.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork(num_channels=3, height=height, width=width)
        self.feature_extractor.to(self.device)

        self.step_count = 0
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

    def _preprocess_images(self, rgb_img: torch.Tensor) -> torch.Tensor:
        """Preprocesses the input RGB image.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).

        Returns:
            torch.Tensor: Preprocessed RGB image, scaled to [0, 1].
        """
        rgb_img = rgb_img / 255.0
        return rgb_img

    def _save_images(self, rgb_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(rgb_img, "shadow_hand_rgb.png")

    def step(self, rgb_img: torch.Tensor, gt_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the image and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """

        rgb_img = self._preprocess_images(rgb_img)

        if self.cfg.write_image_to_file:
            self._save_images(rgb_img)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    # ``torch.cat`` re-materializes the (possibly inference-mode) camera
                    # tensor inside the autograd-enabled block so it can be backpropped.
                    img_input = torch.cat((rgb_img,), dim=-1)
                    self.optimizer.zero_grad()

                    predicted_pose = self.feature_extractor(img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    if self.step_count % 50000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    self.step_count += 1

                    return pose_loss, predicted_pose
        else:
            img_input = torch.cat((rgb_img,), dim=-1)
            predicted_pose = self.feature_extractor(img_input)
            return None, predicted_pose
