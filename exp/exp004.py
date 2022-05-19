""" exp004

Ref:
    * https://www.kaggle.com/code/ammarali32/imc-2022-kornia-loftr-from-0-533-to-0-721
"""
from __future__ import annotations

import csv
import gc
import os
import pdb
import random
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from kornia_moons.feature import draw_LAF_matches
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

IS_NOTEBOOK = "ipykernel" in sys.modules
SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)


def load_matcher(ckpt_path: Path, device: torch.device) -> KF.LoFTR:
    assert ckpt_path.exists(), f"CKPT_PATH => {ckpt_path}"
    print("CKPT PATH => ", ckpt_path)
    matcher = KF.LoFTR(pretrained=None)
    matcher.load_state_dict(torch.load(ckpt_path)["state_dict"])
    matcher = matcher.to(device=device)
    matcher.eval()
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


def load_scaling_factors(scaling_factors_csv_path: Path) -> dict[str, float]:
    assert scaling_factors_csv_path.exists()
    scaling_dict = {}
    with scaling_factors_csv_path.open("r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            scaling_dict[row[0]] = float(row[1])
    return scaling_dict


def read_covisibility_data(covisibility_csv_path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    assert covisibility_csv_path.exists(), f"covisibility_csv_path => {covisibility_csv_path}"
    covisibility_dict = {}
    fundamental_matrix_dict = {}
    with covisibility_csv_path.open("r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # shape of row: (pair, covisibility, fundamental_matrix)
            covisibility_dict[row[0]] = float(row[1])
            fundamental_matrix_dict[row[0]] = np.array([float(v) for v in row[2].split(" ")])
    return covisibility_dict, fundamental_matrix_dict


def make_image_path(image_dir: Path, batch_id: str, image_id: str) -> Path:
    extension = ".png"
    if "train" in str(image_dir):
        extension = ".jpg"
    return image_dir / batch_id / (image_id + extension)


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
    matchered_keypoints0: np.ndarray,
    matchered_keypoints1: np.ndarray,
    ransac_reproj_threshold: float = 0.185,
    confidence: float = 0.9999,
    max_iters: int = 100_000,
) -> np.ndarray:
    """compute fundamental matrix by opencv

    Args:
        image0: an image
        image1: an image
        ransac_reproj_threshold: threshold to determine inlier
        max_iters: the number of iteration
    """
    assert isinstance(matchered_keypoints0, np.ndarray) and isinstance(matchered_keypoints1, np.ndarray)
    # use eight-algorithm
    if len(matchered_keypoints0) < 8:
        return np.zeros((3, 3))

    fundamental_matrix, _ = cv2.findFundamentalMat(
        points1=matchered_keypoints0,
        points2=matchered_keypoints1,
        method=cv2.USAC_MAGSAC,
        # method=cv2.USAC_ACCURATE,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=confidence,
        maxIters=max_iters,
    )
    # inliers = inliers > 0
    assert fundamental_matrix.shape == (3, 3), f"Invalid Shape => {fundamental_matrix.shape}"
    return fundamental_matrix


def make_submission(fundamental_matrix_dict: dict[str, np.ndarray], submission_path: Path) -> None:
    assert submission_path.parent.exists(), f"submission_path => {submission_path}"
    with submission_path.open("w") as f:
        f.write("sample_id,fundamental_matrix\n")
        for sample_id, fundamental_matrix in fundamental_matrix_dict.items():
            f.write(f"{sample_id},{flatten_matrix(fundamental_matrix)}\n")


def plot_LAFs(
    matchered_keypoints0: np.ndarray,
    matchered_keypoints1: np.ndarray,
    inliers: np.ndarray,
    image0: torch.Tensor,
    image1: torch.Tensor,
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


def extract_pairs(covisibility_dict: dict[str, float], threshold: float = 0.1, max_num_pairs: int = 1000) -> list[str]:
    # covisibility_dict: (pair, covisibility)
    pairs = list([pair for pair, covisibility in covisibility_dict.items() if covisibility >= threshold])
    random.shuffle(pairs)
    pairs = pairs[:max_num_pairs]
    return pairs


def get_image(image_data_root: Path, scene: str, image_id: str) -> torch.Tensor:
    assert image_data_root.exists(), f"File Not Found: image_data_root => {image_data_root}"
    image_path = make_image_path(image_data_root, scene, image_id)
    if not image_path.exists():
        raise FileNotFoundError(f"image_path => {image_path}")

    image = load_image(image_path)
    image = scale_image(image)
    image = cast_image_bgr_to_rgb(image)
    image = K.color.rgb_to_grayscale(image)
    return image


def compute_match(
    matcher: KF.LoFTR,
    image_0: torch.Tensor,
    image_1: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    assert (
        len(image_0.shape) == len(image_1.shape) == 4
    ), f"image_0.shape => {image_0.shape}, image_1.shape => {image_1.shape}"

    with torch.inference_mode():
        correspondences = matcher({"image0": image_0, "image1": image_1})

    matchered_keypoints0 = correspondences["keypoints0"].cpu().numpy()
    matchered_keypoints1 = correspondences["keypoints1"].cpu().numpy()
    return matchered_keypoints0, matchered_keypoints1


def parse_fundamental_matrix(fundamental_matrices_str: list[str], sample_ids: list[str]) -> dict[str, np.ndarray]:
    assert isinstance(fundamental_matrices_str, list) and isinstance(
        fundamental_matrices_str[0], str
    ), f"fundamental_matrices_str[0] => {fundamental_matrices_str[0]}"
    predictions = {}
    for sample_id, fundamental_matrix_str in zip(sample_ids, fundamental_matrices_str):
        predictions[sample_id] = np.array([float(v) for v in fundamental_matrix_str.split(" ")]).reshape(3, 3)
    return predictions


class Gt(NamedTuple):
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray


def load_calibration(calibration_csv_path: Path) -> dict[str, Gt]:
    assert calibration_csv_path.exists(), f"Not Found: calibration_csv_path => {calibration_csv_path}"
    calib_dict: dict[str, Gt] = {}
    with calibration_csv_path.open("r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(" ")]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(" ")]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(" ")])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)
    return calib_dict


def decompose_fundamental_matrix_with_intrinsics(
    fundamental_matrix: np.ndarray, K0: np.ndarray, K1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the fundamental matrix into R and T, given the intrinsics.
    Args:
        fundamental_matrix: 基礎行列
        K0: calibration or intrinsics matrix of image 0
        K1: calibration or intrinsics matrix of image 1

    Returns:
        R_a: rotation matrix
        R_b: rotation matrix
        T:
    """
    # This fundamentally reimplements this function:
    # https://github.com/opencv/opencv/blob/be38d4ea932bc3a0d06845ed1a2de84acc2a09de/modules/calib3d/src/five-point.cpp#L742
    # This is a pre-requisite of OpenCV's recoverPose:
    # https://github.com/opencv/opencv/blob/be38d4ea932bc3a0d06845ed1a2de84acc2a09de/modules/calib3d/src/five-point.cpp#L559
    # Instead of the cheirality check with correspondences,
    # we keep and evaluate the different hypotheses downstream, and pick the best one.
    # Note that our metric does not care about the sign of the translation vector,
    # so we only need to evaluate the two rotation matrices.
    assert fundamental_matrix.shape == (3, 3), f"Invalid shape => {fundamental_matrix.shape}"
    assert fundamental_matrix.shape[1] == K0.shape[0]

    E = np.matmul(K1.T, np.matmul(fundamental_matrix, K0))

    u, s, vh = np.linalg.svd(
        E,
    )
    if np.linalg.det(u) < 0:
        u *= -1
    if np.linalg.det(vh) < 0:
        vh *= -1

    w = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R_a = np.matmul(u, np.matmul(w, vh))
    R_b = np.matmul(u, np.matmul(w.T, vh))
    T = u[:, -1]
    return R_a, R_b, T


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """四元数: 回転を表す"""

    M = np.asarray(matrix, dtype=np.float64)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.vstack(
        [
            np.asarray([m00 - m11 - m22, 0.0, 0.0, 0.0]),
            np.asarray([m01 + m10, m11 - m00 - m22, 0.0, 0.0]),
            np.asarray([m02 + m20, m12 + m21, m22 - m00 - m11, 0.0]),
            np.asarray([m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]),
        ]
    )

    K /= 3.0

    # The quaternion is the eigenvecor of K that corresponds to the largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def compute_error_for_one_example(
    q_gt: np.ndarray, T_gt: np.ndarray, q: np.ndarray, T: np.ndarray, scale: float, eps: float = 1e-15
) -> tuple[np.floating, np.floating]:
    """Compute the error metric for a single example.

    The function returns two errors, over rotation and translation.
    These are combined at different thresholds by 'compute_maa' in order to compute the mean Average Accuracy.

    Args:
        q_gt: ground truth of quaternion
        T_gt  ground truth of translation
        q: quaternion, that represents rotation
        T: translation
        scale:
        eps:

    Returns:
        error_q * 180 / np.pi:
        error_t:

    """
    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    error_q = np.arccos(1 - 2 * loss_q)

    # Apply the scale factor for this scene
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    error_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return error_q * 180 / np.pi, error_t


def compute_maa(
    error_q: list[np.floating], error_t: list[np.floating], thresholds_q: np.ndarray, thresholds_t: np.ndarray
) -> tuple[np.floating, np.ndarray, np.ndarray, np.ndarray]:
    assert len(error_q) == len(error_t)

    error_q_np = np.asarray(error_q)
    error_t_np = np.asarray(error_t)

    acc: list[np.floating] = []
    acc_q: list[np.floating] = []
    acc_t: list[np.floating] = []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(error_q_np < th_q, error_t_np < th_t)).sum() / len(error_q_np)]
        acc_q += [(error_q_np < th_q).sum() / len(error_q_np)]
        acc_t += [(error_t_np < th_t).sum() / len(error_t_np)]
    return np.mean(acc), np.asarray(acc), np.asarray(acc_q), np.asarray(acc_t)


def compute_validation_maa(
    predict_df: pd.DataFrame,
    scaling_dict: dict[str, float],
    threshold_q: np.ndarray,
    threshold_t: np.ndarray,
    image_data_root: Path,
    eps: float = 1e-15,
) -> tuple[np.floating, dict[str, np.floating], dict[str, dict[str, np.floating]], dict[str, dict[str, np.floating]]]:
    assert "sample_id" in predict_df.columns and "fundamental_matrix" in predict_df.columns

    predictions = parse_fundamental_matrix(
        fundamental_matrices_str=predict_df["fundamental_matrix"].to_numpy().tolist(),
        sample_ids=predict_df["sample_id"].to_numpy().tolist(),
    )

    scenes: list[str] = []
    for prediciton in predictions.keys():
        _, scene, _ = prediciton.split(";")
        if scene in scenes:
            continue
        if "/images" in scene:
            scene = scene.split("/")[0]
        scenes += [scene]

    calibration_dict: dict[str, dict[str, Gt]] = {
        scene: load_calibration(image_data_root / scene / "calibration.csv") for scene in scenes
    }

    errors_dict_q: dict[str, dict[str, np.floating]] = {scene: {} for scene in scenes}
    errors_dict_t: dict[str, dict[str, np.floating]] = {scene: {} for scene in scenes}
    for sample_id, predicted_fundamental_matrix in tqdm(predictions.items(), total=len(predictions)):
        _, scene, pair = sample_id.split(";")
        if "/images" in scene:
            scene = scene.split("/")[0]
        image_id_0, image_id_1 = pair.split("-")

        gt_0 = calibration_dict[scene][image_id_0]
        gt_1 = calibration_dict[scene][image_id_1]
        K0, R0_gt, T0_gt = gt_0.K, gt_0.R, gt_0.T.reshape((3, 1))
        K1, R1_gt, T1_gt = gt_1.K, gt_1.R, gt_1.T.reshape((3, 1))

        R_pred_a, R_pred_b, T_pred = decompose_fundamental_matrix_with_intrinsics(
            fundamental_matrix=predicted_fundamental_matrix, K0=K0, K1=K1
        )
        q_pred_a = quaternion_from_matrix(R_pred_a)
        q_pred_b = quaternion_from_matrix(R_pred_b)

        dR_gt = np.dot(R1_gt, R0_gt.T)
        dT_gt = (T1_gt - np.dot(dR_gt, T0_gt)).flatten()
        q_gt = quaternion_from_matrix(dR_gt)
        q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

        error_q_a, error_t_a = compute_error_for_one_example(
            q_gt=q_gt, T_gt=dT_gt, q=q_pred_a, T=T_pred, scale=scaling_dict[scene]
        )
        error_q_b, error_t_b = compute_error_for_one_example(
            q_gt=q_gt, T_gt=dT_gt, q=q_pred_b, T=T_pred, scale=scaling_dict[scene]
        )
        assert error_t_a == error_t_b
        errors_dict_q[scene][pair] = np.minimum(error_q_a, error_q_b)
        errors_dict_t[scene][pair] = error_t_a

    maa_per_scene: dict[str, np.floating] = {}
    for scene in scenes:
        maa_per_scene[scene], _, _, _ = compute_maa(
            error_q=list(errors_dict_q[scene].values()),
            error_t=list(errors_dict_t[scene].values()),
            thresholds_q=threshold_q,
            thresholds_t=threshold_t,
        )
    return np.mean(list(maa_per_scene.values())), maa_per_scene, errors_dict_q, errors_dict_t


class ImcDataset(Dataset):
    def __init__(
        self, samples: list[str] | list[list[str]], image_dir_path: Path, is_test: bool, scene: str | None = None
    ) -> None:
        super().__init__()
        # testのときはlist[list[str]]
        self._samples = samples
        self._is_test = is_test
        self._image_dir_path = image_dir_path
        self._scene = scene if scene is not None else "Null"
        if scene is not None and not is_test:
            assert not is_test and "/images" in self._scene, f"scene => {scene}"
        else:
            assert is_test and scene is None

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        if self._is_test:
            row = self._samples[index]
            sample_id, scene, image_0_id, image_1_id = row
        # train & valid
        else:
            scene = self._scene
            pair = self._samples[index]
            if not isinstance(pair, str):
                raise TypeError(f"pair => {pair}, type => {type(pair)}")
            sample_id = f"phototourism;{scene};{pair}"
            image_0_id, image_1_id = pair.split("-")

        image0 = get_image(self._image_dir_path, scene, image_0_id)
        image1 = get_image(self._image_dir_path, scene, image_1_id)
        return image0, image1, sample_id


def validate(
    matcher: KF.LoFTR,
    scaling_dict: dict[str, float],
    data_root: Path,
    save_valid_df_path: Path,
    max_num_pairs: int = 1000,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
) -> None:
    """compute mean Average Accuracy

    Args:
        matcher: kornia.feature.LoFTR
        scaling_dict: scale factor of dict loaded by load_scaling_factors
        save_valid_df_path: save path of predictions of dataframe
        max_num_pairs: the number of validation data

    """
    train_data_root = data_root / "train"

    sample_ids = []
    fundamental_matrices = []
    for scene in scaling_dict.keys():
        covisibility_csv_path = train_data_root / scene / "pair_covisibility.csv"
        covisibility_dict, fundamental_matrix_gt = read_covisibility_data(covisibility_csv_path)

        pairs = extract_pairs(covisibility_dict, threshold=0.1, max_num_pairs=max_num_pairs)
        dataset = ImcDataset(pairs, train_data_root, is_test=False, scene=scene + "/images")
        dataloader = DataLoader(
            dataset, shuffle=False, drop_last=False, pin_memory=True, num_workers=4, batch_size=batch_size
        )

        for batch in tqdm(dataloader, total=len(dataloader)):
            image_0, image_1, sample_id = batch
            image_0, image_1 = image_0.to(device, non_blocking=True), image_1.to(device, non_blocking=True)

            matchered_keypoints0, matchered_keypoints1 = compute_match(
                matcher=matcher, image_0=image_0, image_1=image_1
            )
            fundamental_matrix = make_fundamental_matrix(matchered_keypoints0, matchered_keypoints1)
            if len(sample_id) != 1:
                raise ValueError
            sample_ids.append(sample_id[0])
            fundamental_matrices.append(flatten_matrix(fundamental_matrix))

    valid_df = pd.DataFrame({"sample_id": sample_ids, "fundamental_matrix": fundamental_matrices})
    valid_df.to_csv(save_valid_df_path, index=False)

    threshold_q = np.linspace(1, 10, 10)
    threshold_t = np.geomspace(0.2, 5, 10)

    maa, maa_per_scene, errors_dict_q, _ = compute_validation_maa(
        predict_df=valid_df,
        scaling_dict=scaling_dict,
        threshold_q=threshold_q,
        threshold_t=threshold_t,
        image_data_root=train_data_root,
    )

    for scene, cur_maa in maa_per_scene.items():
        print(f"Scene {scene} ({len(errors_dict_q[scene])} pairs), mAA = {cur_maa:.05f}")
    print(f"mAA = {maa}")

    results_path = save_valid_df_path.parent / "results-maa-per-scene.csv"
    with results_path.open("w") as f:
        f.write("scene,maa\n")
        for scene, maa in maa_per_scene.items():
            f.write(f"{scene},{float(maa):.8f}\n")


def infer(
    matcher: KF.LoFTR,
    test_samples: list[list[str]],
    test_image_dir: Path,
    batch_size: int = 1,
    device: torch.device = torch.device("cuda"),
) -> dict[str, np.ndarray]:
    """
    Args:
        matcher: instance of kornia.feature.LoFTR
        test_samples: test samples by load_test_samples
        test_image_dir: Path instance of test image dir

    Return:
        fundamental_matrix_dict: {sample_id: fundamental_matrix}

    """
    fundamental_matrix_dict = {}
    dataset = ImcDataset(test_samples, test_image_dir, is_test=True)
    dataloader = DataLoader(
        dataset, shuffle=False, drop_last=False, pin_memory=True, num_workers=4, batch_size=batch_size
    )
    pbar = tqdm(dataloader, total=len(dataloader))
    for batch in pbar:
        image_0, image_1, sample_id = batch
        if isinstance(sample_id, list) and len(sample_id) == 1:
            sample_id = sample_id[0]
        image_0, image_1 = image_0.to(device, non_blocking=True), image_1.to(device, non_blocking=True)
        matchered_keypoints0, matchered_keypoints1 = compute_match(matcher=matcher, image_0=image_0, image_1=image_1)
        fundamental_matrix_dict[sample_id] = make_fundamental_matrix(matchered_keypoints0, matchered_keypoints1)

    return fundamental_matrix_dict


def main() -> None:
    exp_name = "exp004"
    is_valid = not IS_NOTEBOOK
    max_num_pairs = 1000
    # max_num_pairs = 5

    input_path = Path("../input") if IS_NOTEBOOK else Path("./input")
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

    if is_valid:
        scaling_dict = load_scaling_factors(data_root / "train" / "scaling_factors.csv")
        save_valid_df_path = output_dir / "valid_df.csv"
        validate(
            matcher=matcher,
            scaling_dict=scaling_dict,
            data_root=data_root,
            save_valid_df_path=save_valid_df_path,
            max_num_pairs=max_num_pairs,
        )

    fundamental_matrix_dict = infer(matcher=matcher, test_samples=test_samples, test_image_dir=test_image_dir)
    make_submission(fundamental_matrix_dict, submission_path)


if __name__ == "__main__":
    main()
