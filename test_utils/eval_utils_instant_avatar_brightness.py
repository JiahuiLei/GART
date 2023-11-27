from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch import nn
import torch
from torch.cuda.amp import custom_fwd
from kornia.enhance.adjust import adjust_brightness_accumulative as kornia_aba


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)  # 1,3,H,W
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)  # 1,3,H,W

        # ! Adjust the brightness here to match the brightness of the ground truth
        # ! here assume a point is pure bright as bg, exclude from the calculation
        # directly enumerate all possible brightness
        br_list = torch.linspace(0.0, 2.0, 100, device=rgb.device)
        pred_msk = (~(rgb == 1.0).all(dim=1, keepdim=True)).expand_as(rgb)
        psnr_list = []
        psnr_best = 0.0
        best_rgb = rgb.clone()
        for br in br_list:
            adj_image = kornia_aba(rgb, br)
            _pred = rgb.clone()
            _pred[pred_msk] = adj_image[pred_msk]
            _metric = self.psnr(_pred, rgb_gt)
            psnr_list.append(float(_metric))
            if _metric > psnr_best:
                psnr_best = _metric
                best_rgb = _pred.clone()
        rgb = best_rgb
        
        # # debug
        # from matplotlib import pyplot as plt

        # plt.plot(br_list.detach().cpu().numpy(), psnr_list)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }
