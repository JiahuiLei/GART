# Use InsAV to process in the wild video, then load it

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
import glob


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }


class Dataset(Dataset):
    # from instant avatar
    def __init__(
        self,
        data_root="data/people_snapshot_public_instant_avatar_processed",
        video_name="male-3-casual",
        split="train",
        image_zoom_ratio=1.0,
        start_end_skip=None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name

        if start_end_skip is not None:
            start, end, skip = start_end_skip
        else:
            # raise NotImplementedError("Must specify, check the end+1")
            if split == "train":
                start, end, skip = 0, 41+1, 1
            elif split == "val":
                start, end, skip = 41, 42+1, 1
            elif split == "test":
                start, end, skip = 42, 51+1, 1

        self.image_zoom_ratio = image_zoom_ratio

        root = osp.join(data_root, video_name)

        camera = np.load(osp.join(root, "cameras.npz"))
        K = camera["intrinsic"]
        T_wc = np.linalg.inv(camera["extrinsic"])
        assert np.allclose(T_wc, np.eye(4))

        height = camera["height"]
        width = camera["width"]

        self.downscale = 1.0 / self.image_zoom_ratio
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale
        self.K = K

        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.png"))[start:end:skip]

        pose_fn = osp.join(root, "poses_optimized.npz")
        smpl_params = load_smpl_param(pose_fn)
        smpl_params["body_pose"] = smpl_params["body_pose"][start:end:skip]
        smpl_params["global_orient"] = smpl_params["global_orient"][start:end:skip]
        smpl_params["transl"] = smpl_params["transl"][start:end:skip]
        self.smpl_params = smpl_params

        # # ! debug
        # pose_fn = osp.join(root, "poses","train.npz")
        # smpl_params = load_smpl_param(pose_fn)
        # smpl_params["body_pose"] = smpl_params["body_pose"][start:end:skip]
        # smpl_params["global_orient"] = smpl_params["global_orient"][start:end:skip]
        # smpl_params["transl"] = smpl_params["transl"][start:end:skip]
        # self.smpl_params = smpl_params

        # cache the images
        self.img_buffer, self.msk_buffer = [], []
        for idx in tqdm(range(len(self.img_lists))):
            img = cv2.imread(self.img_lists[idx])[..., ::-1]
            # msk = np.load(self.msk_lists[idx])
            msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE)
            if self.downscale > 1:
                img = cv2.resize(
                    img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )
                msk = cv2.resize(
                    msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale
                )

            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32) / 255.0
            # apply mask
            # always white
            bg_color = np.ones_like(img).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None])
            self.img_buffer.append(img)
            self.msk_buffer.append(msk)
        return

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = self.img_buffer[idx]
        msk = self.msk_buffer[idx]

        pose = self.smpl_params["body_pose"][idx].reshape((23, 3))
        pose = np.concatenate([self.smpl_params["global_orient"][idx][None], pose], 0)

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk,
            "K": self.K.copy(),
            "smpl_beta": self.smpl_params["betas"][0],  # ! use the first beta!
            "smpl_pose": pose,
            "smpl_trans": self.smpl_params["transl"][idx],
            "idx": idx,
        }

        meta_info = {
            "video": self.video_name,
        }
        viz_id = f"video{self.video_name}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info


if __name__ == "__main__":
    dataset = Dataset(
        data_root="../data/insav_wild", video_name="aist_gBR_sBM_c01_d05_mBR1_ch06"
    )
    ret = dataset[0]
