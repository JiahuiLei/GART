import sys, os, os.path as osp

sys.path.append(osp.dirname(osp.abspath(__file__)))

from ubcfasion_perframe import Dataset as UBCFasionDataset
from instant_avatar_people_snapshot import Dataset as InstantAvatarDataset
from zju_mocap import Dataset as ZJUDataset, get_batch_sampler
from dog_demo import Dataset as DogDemoDataset
from data_provider import RealDataOptimizablePoseProviderPose, DatabasePoseProvider
from instant_avatar_wild import Dataset as InstantAvatarWildDataset
import logging
import numpy as np
import torch


def prepare_real_seq(
    seq_name,
    dataset_mode,
    split="train",
    image_zoom_ratio=0.5,
    balance=False,
    ins_avt_wild_start_end_skip=None,
):
    logging.info("Prepare real seq: {}".format(seq_name))
    # * Get dataset
    if dataset_mode == "ubcfashion":
        dataset = UBCFasionDataset(
            data_root="./data/ubcfashion/",
            video_list=[seq_name],
            image_zoom_ratio=image_zoom_ratio,
            start_end_skip=ins_avt_wild_start_end_skip,
        )
    elif dataset_mode == "people_snapshot":
        dataset = InstantAvatarDataset(
            noisy_flag=False,
            data_root="./data/people_snapshot/",
            video_name=seq_name,
            split=split,
            image_zoom_ratio=image_zoom_ratio,
        )
        print("Load Instant Avatar processed PeopleSnapshot")
    elif dataset_mode == "zju":
        dataset = ZJUDataset(
            data_root="./data/zju_mocap",
            video_name=seq_name,
            split=split,
            image_zoom_ratio=image_zoom_ratio,
        )
    elif dataset_mode == "instant_avatar_wild":
        # assert image_zoom_ratio == 1.0, "Check! in the wild data should use 1.0"
        if image_zoom_ratio != 1.0:
            logging.warning(
                f"Check! in the wild data should use 1.0, but got {image_zoom_ratio}"
            )
        dataset = InstantAvatarWildDataset(
            data_root="./data/insav_wild",
            video_name=seq_name,
            split=split,
            image_zoom_ratio=image_zoom_ratio,
            start_end_skip=ins_avt_wild_start_end_skip,
        )
    elif dataset_mode == "dog_demo":
        dataset = DogDemoDataset(data_root="./data/dog_data_official/", video_name=seq_name)
    else:
        raise NotImplementedError("Unknown mode: {}".format(dataset_mode))

    # prepare an optimizable data provider
    optimizable_data_provider = RealDataOptimizablePoseProviderPose(
        dataset,
        balance=balance,
    )
    return optimizable_data_provider, dataset
