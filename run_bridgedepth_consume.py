from __future__ import division, print_function

import argparse
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

# Ensure repo root is in sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
sys.path.append("core")

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from allegroai import FrameGroup, SingleFrame, StorageManager
from bridgedepth.bridgedepth import BridgeDepth
from torch.amp import autocast  # new AMP API # type: ignore

import depth_estimation.utils.clearml_helpers.dataset_validation as validation
from depth_estimation.scripts.run_evaluation import get_version_by_id
from depth_estimation.utils import image_io
from depth_estimation.utils.clearml_helpers.frames import frame_group_iterator
from depth_estimation.utils.clearml_helpers.metadata import (
    TypedFrameGroupMetadata,
    TypedFrameMetadataEvalModel,
)
from depth_estimation.utils.multithreading import BlockingMaxNConcurrentWorkers
from depth_estimation.utils.path_resolution import (
    local_path_to_s3_path,
    local_path_to_s3_preview_uri,
)
from depth_estimation.utils.streaming import *
from depth_estimation.utils.timing import timer
from depth_estimation.utils.visualization_utils import colorize_depth_image


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class RuntimeConfig:
    scale: float = 1.0  # downscale if GPU memory is tight
    pad_divis: int = 32  # padding multiple for the model
    max_workers: int = 20
    max_queue: int = 500
    max_frames: int = 0  # 0 = no limit
    force_recomputes: bool = True  # re-run even if outputs exist


def log_param_counts(model: torch.nn.Module) -> None:
    """Print total and trainable parameter counts in millions."""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total number of parameters: {total:.2f}M")
    print(f"Total number of trainable parameters: {trainable:.2f}M")


def to_3ch_gray(img: np.ndarray) -> np.ndarray:
    """Ensure HxW -> HxWx3 grayscale triplicate; pass-through if already 3 channels."""
    if img.ndim == 2:
        return np.repeat(img[..., None], 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    return img


def safe_disparity(disp: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Replace non-positive disparities to avoid division by zero."""
    d = disp.clone().float()
    d[d <= 0] = eps
    return d


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Strictly load checkpoint, handling optional 'state_dict' nesting and 'module.' prefix."""
    assert ckpt_path.endswith(".pth") and os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    logging.info("Loading checkpoint...")
    logging.info(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    fixed = {}
    for k, v in checkpoint.items():
        fixed[k if k.startswith("module.") else "module." + k] = v
    model.load_state_dict(fixed, strict=True)
    logging.info("Done loading checkpoint")


def load_and_preprocess_pair(
    path_left: str, path_right: str, device: torch.device, scale: float, pad_divis: int
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Read two images, optional resize, convert to 3ch, to tensor, pad to multiple."""
    img0_np = imageio.imread(path_left)  # HxW (uint8) infra
    img1_np = imageio.imread(path_right)

    orig_h, orig_w = img0_np.shape[:2]
    if scale != 1.0:
        img0_np = cv2.resize(img0_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img1_np = cv2.resize(img1_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    img0_np = to_3ch_gray(img0_np)
    img1_np = to_3ch_gray(img1_np)

    img0 = torch.from_numpy(img0_np).float().permute(2, 0, 1).unsqueeze(0).to(device)
    img1 = torch.from_numpy(img1_np).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # W_MAX = 1024
    # scale = W_MAX / orig_w
    # img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    # img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None, interpolation=cv2.INTER_LINEAR)
    # H, W = img1.shape[:2]
    return img0, img1, (orig_h, orig_w)


@torch.inference_mode()
def run_model(model: torch.nn.Module, img0: torch.Tensor, img1: torch.Tensor, use_amp: bool) -> Tuple[torch.Tensor, float]:
    """
    Expects img0 and img1 as torch.FloatTensors shaped [1, 3, H, W] on the correct device.
    Returns (disp_hw_cpu_tensor, elapsed_seconds).
    """
    assert isinstance(img0, torch.Tensor) and isinstance(img1, torch.Tensor), "img0/img1 must be torch.Tensors"
    assert img0.ndim == 4 and img1.ndim == 4, f"Expected [1,3,H,W], got {img0.shape} and {img1.shape}"

    sample = {
        "img1": img0.contiguous().float(),  # hack the name there
        "img2": img1.contiguous().float(),  # hack the name there
    }

    with autocast(device_type="cuda", enabled=bool(use_amp)):
        with timer(verbose=False) as t:
            results_dict = model(sample)
            disp = results_dict["disp_pred"]
    if disp.ndim == 4 and disp.shape[1] == 1:
        disp = disp[:, 0]  # [B,1,H,W] -> [B,H,W]
    if disp.ndim == 3:
        disp = disp[0]  # [B,H,W]   -> [H,W]
    disp = disp.float().clamp_min(1e-3).cpu()  # keep as torch.Tensor for downstream code

    return disp, t.duration


def save_and_upload_result(
    frame_group,
    frame_name: str,
    disparity: torch.Tensor,
    outpath: str,
    depth_factor_mm: float,
    inference_time_s: float,
):
    """
    Convert disparity to depth (uint16, millimeters), save outputs, upload, and attach to frame group.
    """
    disp_safe = safe_disparity(disparity)
    # Depth(mm) = (f*b in mm·px) / disp(px)  -> multiplied by 1000 to store as uint16 (mm)
    depth_mm = (depth_factor_mm / disp_safe).clamp(min=0.0, max=65535.0)
    depth_mm *= 1000.0
    depth_u16 = depth_mm.to(torch.uint16)

    preview_img = colorize_depth_image(depth_u16)
    preview_path = outpath.replace(".png", "_preview.png")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    image_io.save_depth(image=depth_u16.cpu().numpy(), path=outpath)
    image_io.save_infra(preview_img, preview_path)

    StorageManager.upload_file(local_file=outpath, remote_url=local_path_to_s3_path(outpath), wait_for_upload=True, retries=100)
    StorageManager.upload_file(local_file=preview_path, remote_url=local_path_to_s3_path(preview_path), wait_for_upload=True, retries=100)

    meta_obj = TypedFrameMetadataEvalModel(task_id="task.id", inference_time=inference_time_s, config={})
    # meta_dict = meta_obj.model_dump() if hasattr(meta_obj, "model_dump") else meta_obj.dict()
    meta_dict = meta_obj.to_metadata()
    frame = SingleFrame(
        source=local_path_to_s3_path(outpath),
        width=depth_u16.shape[1],
        height=depth_u16.shape[0],
        preview_uri=local_path_to_s3_preview_uri(preview_path),
        metadata=meta_dict,
    )
    frame_group[frame_name] = frame


def main():

    
    # task = create_clearml_task(
    #     task_name="Test ClearML Rerun",
    #     tag="blackwell-bridgedepth",
    #     setup_script=". /home/perception/miniconda3/bin/activate && conda activate bridgedepth",
    #     python_binary="/home/perception/miniconda3/envs/bridgedepth/bin/python3",
    #     python_path="/opt/ros/noetic/lib/python3/dist-packages:/home/perception/dev",
    #     pip_version="25.0.1",
    # )
    # task.execute_remotely(
    #     queue_name="perception-pointclaude",
    #     exit_process=True,
    # )
    dataset = get_version_by_id(dataset_id="4b647cef777a4ad5a27b185005c66ede", version_id="d883698e298a4b35ab8a818f37c91510")
    frame_name = "BridgeDepth"
    force_recomputes = True
    filter_query = None  # eg. filter_query = 'meta.bagname="underworld_stairs_pt2_2025-05-13-14-03-21_anymal_record"'
    parser = argparse.ArgumentParser()
    checkpoint_path = StorageManager.download_file("s3://172.16.0.71:10050/stereo-datasets-gt/BridgeDepth/bridge_rvc_pretrain.pth")
    parser.add_argument("--model_name", choices=["rvc", "rvc_pretrain", "eth3d_pretrain", "middlebury_pretrain"], default="rvc_pretrain")
    parser.add_argument("--checkpoint_path", default=checkpoint_path, type=str)
    parser.add_argument("--out_dir", default="demo_output", type=str, help="the directory to save results")
    parser.add_argument("--z_far", default=5, type=float, help="max depth to clip in point cloud")
    parser.add_argument("--get_pc", type=int, default=1, help="save point cloud output")
    parser.add_argument("--use_amp", default=True, help="use mixed precision")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")

    # Build model & load weights
    model = BridgeDepth.from_pretrained(args.checkpoint_path)
    model = model.to(torch.device("cuda"))
    model.eval()
    log_param_counts(model)

    # Runtime config
    cfg = RuntimeConfig()
    elapsed_list: list[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = BlockingMaxNConcurrentWorkers(max_workers=cfg.max_workers, max_queue=cfg.max_queue)

    def compute_bridgedepth(frame_group: FrameGroup) -> Optional[FrameGroup]:
        assert frame_group["Infra1"].source
        assert frame_group["Infra2"].source
        infra1 = StorageManager.download_file(frame_group["Infra1"].source)
        infra2 = StorageManager.download_file(frame_group["Infra2"].source)
        assert infra1
        assert infra2

        outpath = infra1.replace("/Infra1/", f"/{frame_name}/")

        # Prep
        img0, img1, orig_hw = load_and_preprocess_pair(infra1, infra2, device=device, scale=cfg.scale, pad_divis=cfg.pad_divis)

        # Inference
        disp, elapsed = run_model(model, img0, img1, use_amp=args.use_amp)
        elapsed_list.append(elapsed)

        # Persist (threaded)
        metadata = TypedFrameGroupMetadata.model_validate(frame_group.metadata)
        depth_factor_mm = float(metadata.baseline)
        save_and_upload_result(
            frame_group=frame_group,
            frame_name=frame_name,
            disparity=disp,
            outpath=outpath,
            depth_factor_mm=depth_factor_mm,
            inference_time_s=elapsed,
        )
        return frame_group

    torch.set_num_threads(20)
    torch.set_num_interop_threads(20)
    torch.set_grad_enabled(False)
    consume(
        pipe(
            frame_group_iterator(dataset, filter_query=filter_query),
            prefetch(5000),
            # skip frames already computed
            filter_by((lambda fg: (frame_name not in fg) or force_recomputes)),
            report_progress(desc=f"Found with missing {frame_name} frame or Force_recomputes = True", unit="Frame Groups", position=0),
            limit_items(cfg.max_frames),
            # check preconditions
            filter_by(validation.check_has_frames(frame_names=["Infra1", "Infra2"], verbose=False)),
            report_progress(desc=f"Found Infra frame", unit="Frame Groups", position=0),
            # download infra1 and infra 2
            parallel_map(workers=15, chunk_size=1, fn=filter_by(check_all(ensure_infra1_downloaded, ensure_infra2_downloaded))),
            report_progress(desc="Downloaded Infra1 and Infra2", unit="Frame Groups", position=1),
            prefetch(300),
            # compute bridgedepth
            filter_by(compute_bridgedepth),
            report_progress(desc="Computed Bridgedepth", unit="Frame Groups", position=2),
            prefetch(5000),
            update_dataset_chunked(dataset),
            report_progress(desc="Updated Dataset", unit="Frame Groups", position=3),
        )
    )

    workers.wait_for_all_jobs()
    if len(elapsed_list) > 0:
        avg_runtime = float(np.mean(elapsed_list))
        print(f"Average runtime: {avg_runtime:.3f}s, {1.0/avg_runtime:.2f} FPS")


if __name__ == "__main__":
    main()
