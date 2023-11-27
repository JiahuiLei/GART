# load each frame in the video with: 1. rgb image, mask, K, template model vtx in camera coordinate

# per-frame version

from torch.utils.data import Dataset
import logging
import json
import os
import numpy as np
from os.path import join
import os.path as osp
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
import imageio
import cv2
from plyfile import PlyData
from tqdm import tqdm
from transforms3d.euler import euler2mat
from pytorch3d.transforms import matrix_to_axis_angle
import torch
from pycocotools import mask as masktool


def read_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def get_camera(camera_path):
    camera = read_pickle(camera_path)
    K = np.zeros([3, 3])
    K[0, 0] = camera["camera_f"][0]
    K[1, 1] = camera["camera_f"][1]
    K[:2, 2] = camera["camera_c"]
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera["camera_k"]
    camera = {"K": K, "R": R, "T": T, "D": D}
    return camera


def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def crop_to_center(K, img_list):
    H, W = img_list[0].shape[:2]
    Cx, Cy = float(K[0, 2]), float(K[1, 2])
    half_size_x = int(np.floor(min(Cx, W - Cx)))
    half_size_y = int(np.floor(min(Cy, H - Cy)))
    if Cx < W - Cx:
        ret = [img[:, : 2 * half_size_x, :] for img in img_list]
    else:
        ret = [img[:, -2 * half_size_x :, :] for img in img_list]
    if Cy < H - Cy:
        ret = [img[: 2 * half_size_y, :, :] for img in ret]
    else:
        ret = [img[-2 * half_size_y :, :, :] for img in ret]
    new_K = K.copy()
    new_K[0, 2] = half_size_x - 0.5
    new_K[1, 2] = half_size_y - 0.5
    return new_K, ret


class Dataset(Dataset):
    def __init__(
        self,
        data_root="data/ubcfashion",
        video_list=["91+20mY7UJS"],
        image_zoom_ratio=1.0,
        start_end_skip=None,
        # mask_bkgd=True,
        # white_bkgd=True,
    ) -> None:
        super().__init__()
        
        # raise NotImplementedError("This dataset is not ready yet, check the end+1")

        self.data_root = data_root
        self.image_root = join(data_root, "train_frames")
        self.mask_root = join(data_root, "train_mask")
        self.pose_root = join(data_root, "train_smpl")

        if start_end_skip is None:
            start_end_skip = [0, 100000000, 1]
        self.start_end_skip = start_end_skip

        self.video_list = video_list
        if len(self.video_list) == 1 and self.video_list[0] == "all":
            self.video_list = [d for d in os.listdir(self.image_root)]
        self.image_zoom_ratio = image_zoom_ratio
        # cache
        self.data_list = self.cache(self.video_list)
        return

    def cache(self, dir_list):
        ret = []
        for dir in dir_list:
            frames = os.listdir(join(self.image_root, dir))
            frames.sort()
            pose_fn = join(self.pose_root, dir + ".npz")
            pose_data = np.load(pose_fn)
            smpl_shape = pose_data["pred_shape"].mean(0)  # Use the average shape
            smpl_pose_list, smpl_global_trans = (
                pose_data["pred_rotmat"],
                pose_data["pred_trans"],
            )
            focal, center = pose_data["img_focal"], pose_data["img_center"]

            mask_fn = join(self.mask_root, dir + ".npy")
            masks = np.load(mask_fn, allow_pickle=True)
            masks = [m for m in masks]

            K = np.eye(3)
            K[0, 0], K[1, 1] = focal, focal
            K[0, 2], K[1, 2] = center[0], center[1]
            start, end, skip = self.start_end_skip
            for fid in tqdm(frames[start:end:skip]):
                fid = fid.split(".")[0]
                img = (
                    imageio.imread(join(self.image_root, dir, fid + ".png")).astype(
                        np.float32
                    )
                    / 255.0
                )
                mask = masktool.decode(masks[int(fid)].copy())
                mask = mask.astype(np.bool).astype(np.float32)

                assert (
                    img.shape[0] - center[1] * 2
                ) < 0.5, "Only support optical axis at center"
                assert (
                    img.shape[1] - center[0] * 2
                ) < 0.5, "Only support optical axis at center"
                H, W = int(img.shape[0] * self.image_zoom_ratio), int(
                    img.shape[1] * self.image_zoom_ratio
                )
                if self.image_zoom_ratio != 1.0:
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                ret_K = K.copy().astype(np.float32)
                ret_K[:2] = ret_K[:2] * self.image_zoom_ratio

                fid = int(fid)
                rot = torch.from_numpy(smpl_pose_list[fid])
                pose = matrix_to_axis_angle(rot).numpy()
                ret.append(
                    {
                        "video": dir,
                        "frame": fid,
                        # data
                        "rgb": img,  # H,W,3
                        "mask": mask,  # H,W
                        "K": ret_K,
                        # pose
                        "smpl_beta": smpl_shape.copy(),  # 10
                        "smpl_pose": pose,  # 24,3
                        "smpl_trans": smpl_global_trans[fid, 0],  # 3
                    }
                )
        return ret

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        meta_info = {
            "video": data["video"],
            "frame": data["frame"],
        }
        viz_id = f"video{data['video']}_frame{data['frame']}_dataidx{index}"
        meta_info["viz_id"] = viz_id
        ret = data
        return ret, meta_info


if __name__ == "__main__":
    # Long hard 81FyMPk-WIS
    # tight easy 91+20mY7UJS
    dataset = Dataset(
        data_root="../data/ubcfashion",
    )
    dbg, meta_info = dataset[0]
    print(dbg.keys())
