# take the instant avatar format
# ! Warning, must use the pre-processed People Snapshot!!
import numpy as np
import os.path as osp
import numpy as np
import glob
import imageio
from tqdm import tqdm
from transforms3d.euler import euler2mat
from pycocotools import mask as masktool

import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


TRAIN_RANGES = {
    "hound_": [[178, 220], [262, 310], [339, 441]],
    "hound": [[22, 60], [130, 144], [150, 223], [250, 441]],
    "french": [[150, 384], [468, 530]],
    "german": [[2, 45], [93, 284], [291, 400]],
    "alaskan": [[65, 90], [153, 185], [255, 325], [350, 440], [630, 667]],
    "pit_bull": [[100, 127], [129, 492], [629, 808]],
    "irish": [
        [90, 127],
        [328, 384],
        [390, 400],
        [430, 470],
        [549, 564],
        [584, 625],
        [881, 1008],
    ],
    "english": [[164, 264], [292, 615], [660, 664]],
    "shiba": [[75, 200], [317, 343], [366, 500]],
    "corgi": [
        [30, 66],
        [150, 190],
        [253, 266],
        [495, 513],
        [573, 588],
        [639, 660],
        [779, 796],
        [829, 860],
    ],
}

TEST_RANGES = {
    "french": [[635, 815]],
    "german": [[421, 550]],
    "alaskan": [[493, 565]],
    "pit_bull": [[493, 628]],
    "hound": [[442, 571]],
    "english": [[678, 883]],
    "irish": [[1136, 1274]],
    "shiba": [[501, 650]],
    "corgi": [[861, 877]],
}


def get_frame_id_list(video_name):
    bounds = TRAIN_RANGES[video_name]
    ids = []
    for b in bounds:
        ids += list(range(b[0], b[1] + 1))
    return ids


def get_test_frame_id_list(video_name, test_size=15):
    bounds = TEST_RANGES[video_name]
    ids = []
    for b in bounds:
        ids += list(range(b[0], b[1] + 1, (b[1] - b[0] + 1) // test_size))
    ids = ids[:test_size]
    return ids


class Dataset(Dataset):
    def __init__(
        self, data_root="data/dog_data", video_name="hound", test=False
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.video_name = video_name
        root = osp.join(data_root, video_name)

        image_dir = osp.join(root, "images")
        pose_dir = osp.join(root, "pred")

        if test:
            id_list = get_test_frame_id_list(video_name)
        else:
            id_list = get_frame_id_list(video_name)

        self.rgb_list, self.mask_list = [], []
        betas_list = []
        self.pose_list, self.trans_list = [], []
        self.K_list = []

        for i in tqdm(id_list):
            img_path = osp.join(image_dir, f"{i:04d}.png")
            msk_path = osp.join(image_dir, f"{i:04d}.npy")
            pose_path = osp.join(pose_dir, f"{i:04d}.npz")
            if not osp.exists(msk_path):
                continue

            rgb = imageio.imread(img_path)
            assert rgb.shape[0] == 512 and rgb.shape[1] == 512
            mask = np.load(msk_path, allow_pickle=True).item()
            mask = masktool.decode(mask)

            pred = dict(np.load(pose_path, allow_pickle=True))
            betas = pred["pred_betas"]
            betas_limbs = pred["pred_betas_limbs"]

            pose = pred["pred_pose"]
            pose = matrix_to_axis_angle(torch.from_numpy(pose)).numpy()[0].reshape(-1)
            trans = pred["pred_trans"][0]
            focal = pred["pred_focal"] * 2  # for 512 size image

            K = np.eye(3)
            K[0, 0], K[1, 1] = focal, focal
            K[0, 2], K[1, 2] = 256, 256

            rgb = (rgb[..., :3] / 255).astype(np.float32)
            mask = mask.astype(np.float32)
            # apply mask
            rgb = rgb * mask[..., None] + (1 - mask[..., None])

            self.rgb_list.append(rgb)
            self.mask_list.append(mask)
            betas_list.append(betas)
            self.pose_list.append(np.concatenate([pose, betas_limbs[0]], 0))
            self.trans_list.append(trans)
            self.K_list.append(K)
        # average the beta
        self.betas = np.concatenate(betas_list, 0).mean(0)
        print(f"Loaded {len(self.rgb_list)} frames from {video_name}")
        return

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, idx):
        img = self.rgb_list[idx]
        msk = self.mask_list[idx]
        pose = self.pose_list[idx]

        ret = {
            "rgb": img.astype(np.float32),
            "mask": msk,
            "K": self.K_list[idx].copy(),
            "smpl_beta": self.betas,
            "smpl_pose": pose,
            "smpl_trans": self.trans_list[idx],
            "idx": idx,
        }

        meta_info = {
            "video": self.video_name,
        }
        viz_id = f"video{self.video_name}_dataidx{idx}"
        meta_info["viz_id"] = viz_id
        return ret, meta_info


if __name__ == "__main__":
    dataset = Dataset(data_root="../data/dog_demo")
    ret = dataset[0]
