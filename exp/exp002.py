""" exp002

* make baseline Ref:
    * https://www.kaggle.com/code/ammarali32/imc-2022-kornia-loftr-from-0-533-to-0-721
"""
from __future__ import annotations

import csv
import gc
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import kornia
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from kornia_moons.feature import draw_LAF_matches
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

IS_NOTEBOOK = "ipykernel" in sys.modules


def load_matcher(ckpt_path: Path, device: torch.device) -> KF.LoFTR:
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(ckpt_path)["state_dict"])
    matcher = matcher.to(device=device)
    return matcher


def load_test_samples(test_csv_path: Path) -> list[list[str]]:
    assert test_csv_path.exists()
    test_samples = []
    with test_csv_path.open("r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:  # skip header
                continue
            test_samples += [row]
    return test_samples


def make_image_path(image_dir: Path, batch_id: str, image_id: str) -> Path:
    return image_dir / batch_id / (image_id + ".png")


def flatten_matrix(matrix: np.ndarray, num_digits: int = 8) -> str:
    return " ".join([f"{v:.{num_digits}e}" for v in matrix.flatten()])


def load_image(fname: Path) -> torch.Tensor:
    assert fname.exists(), f"Not Found => {fname}"
    img = read_image(str(fname), mode=ImageReadMode.UNCHANGED)
    return img


def cast_image_bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    return K.color.bgr_to_rgb(image)


def scale_image(image: torch.Tensor) -> torch.Tensor:
    """

    Args:
        image: (channels, h, w)
    """
    # 元の画像スケールに合わせて、両辺が８の倍数になるように調整
    w = int(image.shape[2] // 8 * 8)
    h = int(image.shape[1] // 8 * 8)
    # 正規化
    image = T.functional.resize(image, size=[h, w]).float() / 255.0
    return image


def make_fundamental_matrix(
    matchered_keypoints0: np.ndarray, matchered_keypoints1: np.ndarray
) -> np.ndarray:
    # use eight-algorithm
    if len(matchered_keypoints0) < 8:
        return np.zeros((3, 3))

    fundamental_matrix, inliers = cv2.findFundamentalMat(
        matchered_keypoints0,
        matchered_keypoints1,
        cv2.USAC_MAGSAC,
        0.185,
        0.9999,
        100_000,
    )
    # inliers = inliers > 0
    assert fundamental_matrix.shape == (3, 3), f"Invalid Shape => {fundamental_matrix.shape}"
    return fundamental_matrix


def make_submission(
    fundamental_matrix_dict: dict[str, np.ndarray], submission_path: Path
) -> None:
    with submission_path.open("w") as f:
        f.write("sample_id,fundamental_matrix\n")
        for sample_id, fundamental_matrix in fundamental_matrix_dict.items():
            f.write(f"{sample_id},{flatten_matrix(fundamental_matrix)}\n")


def plot_LAFs(
    matchered_keypoints0: np.ndarray,
    matchered_keypoints1: np.ndarray,
    inliers: np.ndarray,
    image0: torch.Tensor,
    image1: torch.Tensor
) -> None:
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(
            torch.from_numpy(matchered_keypoints0).view(1, -1, 2),
            torch.ones(matchered_keypoints0.shape[0]).view(1, -1, 1, 1),
            torch.ones(matchered_keypoints0.shape[0]).view(1, -1, 1),
        ),
        KF.laf_from_center_scale_ori(
            torch.from_numpy(matchered_keypoints1).view(1, -1, 2),
            torch.ones(matchered_keypoints1.shape[0]).view(1, -1, 1, 1),
            torch.ones(matchered_keypoints1.shape[0]).view(1, -1, 1),
        ),
        torch.arange(matchered_keypoints0.shape[0]).view(-1, 1).repeat(1, 2),
        K.tensor_to_image(image0),
        K.tensor_to_image(image1),
        inliers,
        draw_dict={
            "inlier_color": (0.2, 1, 0.2),
            "tentative_color": None,
            "feature_color": (0.2, 0.5, 1),
            "vertical": False,
        },
    )


def infer(
    matcher: KF.LoFTR, test_samples: list[list[str]], test_image_dir: Path, device: torch.device = torch.device("cuda")
) -> dict[str, np.ndarray]:
    fundamental_matrix_dict = {}
    pbar = tqdm(enumerate(test_samples), total=len(test_samples))
    for i, row in pbar:
        sample_id, batch_id, image_0_id, image_1_id = row
        image_0_path = make_image_path(test_image_dir, batch_id, image_0_id)
        image_1_path = make_image_path(test_image_dir, batch_id, image_1_id)
        image_0 = load_image(image_0_path)
        image_1 = load_image(image_1_path)
        image_0 = scale_image(image_0)
        image_1 = scale_image(image_1)
        image_0 = cast_image_bgr_to_rgb(image_0).to(device=device)
        image_1 = cast_image_bgr_to_rgb(image_1).to(device=device)

        input_dict = {
            "image0": K.color.rgb_to_grayscale(image_0).unsqueeze(0),
            "image1": K.color.rgb_to_grayscale(image_1).unsqueeze(0)
        }
        # expected insput shape of tensor : (BS, 1, H, W)
        with torch.inference_mode():
            correspondences = matcher(input_dict)

        matchered_keypoints0 = correspondences["keypoints0"].cpu().numpy()
        matchered_keypoints1 = correspondences["keypoints1"].cpu().numpy()
        fundamental_matrix_dict[sample_id] = make_fundamental_matrix(matchered_keypoints0, matchered_keypoints1)

    return fundamental_matrix_dict


def main() -> None:
    exp_name = "exp001"

    input_path = Path("./input")
    data_root = input_path / "image-matching-challenge-2022"
    test_csv_path = data_root / "test.csv"
    test_image_dir = data_root / "test_images"
    test_samples = load_test_samples(test_csv_path)

    output_dir = Path("./") if IS_NOTEBOOK else Path("./output") / exp_name
    submission_path = output_dir / "submission.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path("./output/checkpoints/LoFTR/loftr_outdoor.ckpt")
    device = torch.device("cuda")
    matcher = load_matcher(ckpt_path, device=device)

    fundamental_matrix_dict = infer(matcher, test_samples, test_image_dir)
    make_submission(fundamental_matrix_dict, submission_path)


if __name__ == "__main__":
    main()
