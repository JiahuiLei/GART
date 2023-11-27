from matplotlib import pyplot as plt
import torch
import numpy as np
import os, os.path as osp
import shutil
from torch.utils.tensorboard import SummaryWriter
import logging, platform
from datetime import datetime


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def fov2K(fov=90, H=256, W=256):
    if isinstance(fov, torch.Tensor):
        f = H / (2 * torch.tan(fov / 2 * np.pi / 180))
        K = torch.eye(3).repeat(fov.shape[0], 1, 1).to(fov)
        K[:, 0, 0], K[:, 0, 2] = f, W / 2.0
        K[:, 1, 1], K[:, 1, 2] = f, H / 2.0
        return K.clone()
    else:
        f = H / (2 * np.tan(fov / 2 * np.pi / 180))
        K = np.eye(3)
        K[0, 0], K[0, 2] = f, W / 2.0
        K[1, 1], K[1, 2] = f, H / 2.0
        return K.copy()


def get_bbox(mask, padding):
    # mask: H,W, 0-1, get the bbox with padding
    assert mask.ndim == 2
    assert isinstance(mask, torch.Tensor)
    # x is width, y is hight
    xm = mask.sum(dim=0) > 0
    ym = mask.sum(dim=1) > 0

    xl, xr = xm.nonzero().min(), xm.nonzero().max()
    yl, yr = ym.nonzero().min(), ym.nonzero().max()

    xl, xr = max(0, xl - padding), min(mask.shape[1], xr + padding)
    yl, yr = max(0, yl - padding), min(mask.shape[0], yr + padding)

    box = torch.zeros_like(mask)
    box[yl:yr, xl:xr] = 1.0

    return yl, yr, xl, xr


def create_log(log_dir, name, debug=False):
    os.makedirs(osp.join(log_dir, "viz_step"), exist_ok=True)
    backup_dir = osp.join(log_dir, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copyfile(__file__, osp.join(backup_dir, osp.basename(__file__)))
    shutil.copytree("lib_gart", osp.join(backup_dir, "lib_gart"), dirs_exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # backup all notebooks
    os.system(f"cp ./*.ipynb {backup_dir}/")

    # also set the logging to print to terminal and the file
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    configure_logging(osp.join(log_dir, f"{current_datetime}.log"), debug=debug, name=name)
    return writer


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def configure_logging(log_path, debug=False, name="default"):
    """
    https://github.com/facebookresearch/DeepSDF
    """
    logging.getLogger().handlers.clear()
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    logger_handler.addFilter(HostnameFilter())
    formatter = logging.Formatter(
        "| %(hostname)s | %(levelname)s | %(asctime)s | %(message)s   [%(filename)s:%(lineno)d]",
        "%b-%d-%H:%M:%S",
    )
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)
    
    file_logger_handler = logging.FileHandler(log_path)
    
    file_logger_handler.setFormatter(formatter)
    logger.addHandler(file_logger_handler)
