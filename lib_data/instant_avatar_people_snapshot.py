# take the instant avatar format
# ! Warning, must use the pre-processed People Snapshot!!

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

START_END = {
    "female-3-casual": {
        "train": [0, 445, 4],  # start, end skip
        "val": [446, 446, 4],
        "test": [446, 647, 4],
    },
    "female-4-casual": {"train": [0, 335, 4], "val": [335, 335, 4], "test": [335, 523, 4]},
    "female-4-sport": {"train": [1, 336, 4], "val": [336, 336, 8], "test": [336, 524, 4]},
    # "male-2-casual": {
    #     "train": [1, 445, 4],
    #     "val": [445, 445, 8],
    #     "test": [1, 445, 20],
    # },  # ! this is wired, but this is from the original InsAva code
    # "male-2-sport": {
    #     "train": [1, 455, 4],
    #     "val": [455, 455, 8],
    #     "test": [1, 455, 20],
    # },  # ! this is wired, but this is from the original InsAva code
    "male-3-casual": {"train": [0, 455, 4], "val": [456, 456, 4], "test": [456, 675, 4]},
    "male-3-sport": {"train": [1, 230, 2], "val": [230, 230, 6], "test": [230, 460, 6]},
    "male-4-casual": {"train": [0, 659, 6], "val": [660, 660, 6], "test": [660, 872, 6]},
}


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
        noisy_flag,
        data_root="data/people_snapshot_public_instant_avatar_processed",
        video_name="male-3-casual",
        split="train",
        image_zoom_ratio=0.5,
        use_refined_pose=True, # ! Use Instant Avatar refined pose!!
    ) -> None:
        super().__init__()
        self.noisy_flag = noisy_flag
        self.data_root = data_root
        self.video_name = video_name

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

        start = START_END[video_name][split][0]
        end = START_END[video_name][split][1] + 1
        skip = START_END[video_name][split][2]
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.npy"))[start:end:skip]

        # ! we take refine false
        if use_refined_pose:
            if noisy_flag:
                pose_fn = osp.join(root, "poses_noisier", f"anim_nerf_{split}.npz")
            else:
                pose_fn = osp.join(root, "poses", f"anim_nerf_{split}.npz")
            self.smpl_params = load_smpl_param(pose_fn)
        else:
            self.smpl_params = load_smpl_param(osp.join(root, "poses.npz"))
            for k, v in self.smpl_params.items():
                if k != "betas":
                    self.smpl_params[k] = v[start:end:skip]

        # cache the images
        self.img_buffer, self.msk_buffer = [], []
        for idx in tqdm(range(len(self.img_lists))):
            img = cv2.imread(self.img_lists[idx])[..., ::-1]
            msk = np.load(self.msk_lists[idx])
            if self.downscale > 1:
                img = cv2.resize(img, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale)
                msk = cv2.resize(msk, dsize=None, fx=1 / self.downscale, fy=1 / self.downscale)

            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32)
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
            "smpl_beta": self.smpl_params["betas"][0], # ! use the first beta!
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
    dataset = Dataset(data_root="../data/people_snapshot_public_instant_avatar_processed")
    ret = dataset[0]