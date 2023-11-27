import os
import cv2
import torch
import numpy as np
import lpips as lp
from termcolor import cprint, colored
from torch import nn

try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity as compare_ssim

# ! Instant-NVR test_full=True


class Evaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = lp.LPIPS(net="vgg", verbose=False).cuda()
        self.loss_fn.eval()
        for p in self.loss_fn.parameters():
            p.requires_grad_(False)

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def forward(self, rgb, rgb_gt):
        # rgb: B,H,W,3
        assert len(rgb) == 1 and len(rgb_gt) == 1

        img_pred = rgb[0].detach().cpu().numpy()
        img_gt = rgb_gt[0].detach().cpu().numpy()

        # mse = np.mean((img_pred - img_gt)**2)
        # self.mse.append(mse)
        psnr = self.psnr_metric(img_pred.reshape(-1, 3), img_gt.reshape(-1, 3))
        lpips = (
            self.loss_fn(
                torch.tensor(img_pred.transpose((2, 0, 1)), dtype=torch.float, device="cuda")[None],
                torch.tensor(img_gt.transpose((2, 0, 1)), dtype=torch.float, device="cuda")[None],
            )[0]
            .detach()
            .cpu()
            .numpy()
        )

        # breakpoint()
        # ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        ssim = compare_ssim(img_pred, img_gt, channel_axis=2)

        return {
            "psnr": torch.Tensor([float(psnr)]).squeeze(),
            "ssim": torch.Tensor([float(ssim)]).squeeze(),
            "lpips": torch.Tensor([float(lpips)]).squeeze(),
        }


# class Evaluator_instant_nvr:
#     def __init__(self):
#         self.mse = []
#         self.psnr = []
#         self.ssim = []
#         self.lpips = []
#         self.loss_fn = lp.LPIPS(net="vgg", verbose=False).cuda()
#         self.loss_fn.eval()
#         for p in self.loss_fn.parameters():
#             p.requires_grad_(False)

#     def psnr_metric(self, img_pred, img_gt):
#         mse = np.mean((img_pred - img_gt) ** 2)
#         psnr = -10 * np.log(mse) / np.log(10)
#         return psnr

#     def ssim_metric(self, rgb_pred, rgb_gt, batch, epoch=-1):
#         mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
#         H, W = batch["H"].item(), batch["W"].item()
#         mask_at_box = mask_at_box.reshape(H, W)

#         # convert the pixels into an image
#         img_pred = np.zeros((H, W, 3))
#         img_pred[mask_at_box] = rgb_pred
#         img_gt = np.zeros((H, W, 3))
#         img_gt[mask_at_box] = rgb_gt

#         orig_img_pred = img_pred.copy()
#         orig_img_gt = img_gt.copy()

#         if "crop_bbox" in batch:
#             img_pred = fill_image(img_pred, batch)
#             img_gt = fill_image(img_gt, batch)

#         if epoch != -1:
#             result_dir = os.path.join(cfg.result_dir, f"comparison_epoch{epoch}")
#         else:
#             result_dir = os.path.join(cfg.result_dir, "comparison")
#         os.system("mkdir -p {}".format(result_dir))
#         frame_index = batch["frame_index"].item()
#         view_index = batch["cam_ind"].item()
#         cv2.imwrite(
#             "{}/frame{:04d}_view{:04d}.png".format(result_dir, frame_index, view_index),
#             (img_pred[..., [2, 1, 0]] * 255),
#         )
#         cv2.imwrite(
#             "{}/frame{:04d}_view{:04d}_gt.png".format(result_dir, frame_index, view_index),
#             (img_gt[..., [2, 1, 0]] * 255),
#         )

#         # crop the object region
#         x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
#         img_pred = orig_img_pred[y : y + h, x : x + w]
#         img_gt = orig_img_gt[y : y + h, x : x + w]
#         # compute the ssim
#         ssim = compare_ssim(img_pred, img_gt, multichannel=True)

#         return ssim

#     def evaluate(self, output, batch, epoch=-1):
#         rgb_pred = output["rgb_map"][0].detach().cpu().numpy()
#         rgb_gt = batch["rgb"][0].detach().cpu().numpy()

#         if cfg.test_full:
#             mask_at_box = batch["mask_at_box"][0].detach().cpu().numpy()
#             H, W = batch["H"].item(), batch["W"].item()
#             mask_at_box = mask_at_box.reshape(H, W)

#             img_pred = np.zeros((H, W, 3))
#             img_pred[mask_at_box] = rgb_pred

#             img_gt = np.zeros((H, W, 3))
#             img_gt[mask_at_box] = rgb_gt

#             if cfg.eval_part != "":
#                 msk = batch["sem_mask"][partnames.index(cfg.eval_part)]
#                 img_pred[~msk] = 0
#                 img_gt[~msk] = 0

#             frame_index = batch["frame_index"].item()
#             view_index = batch["cam_ind"].item()
#             if epoch != -1:
#                 result_dir = os.path.join(cfg.result_dir, f"comparison_epoch{epoch}")
#             else:
#                 result_dir = os.path.join(cfg.result_dir, "comparison")

#             if not cfg.fast_eval:
#                 if not os.path.exists(result_dir):
#                     os.mkdir(result_dir)
#                 cv2.imwrite(
#                     "{}/frame{:04d}_view{:04d}.png".format(result_dir, frame_index, view_index),
#                     (img_pred[..., [2, 1, 0]] * 255),
#                 )
#                 cv2.imwrite(
#                     "{}/frame{:04d}_view{:04d}_gt.png".format(result_dir, frame_index, view_index),
#                     (img_gt[..., [2, 1, 0]] * 255),
#                 )

#             if cfg.dry_run:
#                 return

#             mse = np.mean((img_pred - img_gt) ** 2)
#             self.mse.append(mse)

#             psnr = self.psnr_metric(img_pred.reshape(-1, 3), img_gt.reshape(-1, 3))
#             self.psnr.append(psnr)

#             lpips = (
#                 self.loss_fn(
#                     torch.tensor(img_pred.transpose((2, 0, 1)), dtype=torch.float, device="cuda")[
#                         None
#                     ],
#                     torch.tensor(img_gt.transpose((2, 0, 1)), dtype=torch.float, device="cuda")[
#                         None
#                     ],
#                 )[0]
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )
#             self.lpips.append(lpips)

#             # breakpoint()
#             # ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
#             ssim = compare_ssim(img_pred, img_gt, channel_axis=2)
#             self.ssim.append(ssim)

#             # print(f"mse: {mse}")
#             # print(f"psnr: {psnr}")
#             # print(f"ssim: {ssim}")
#             # print(f"lpips: {lpips}")
#         else:
#             if rgb_gt.sum() == 0:
#                 return

#             mse = np.mean((rgb_pred - rgb_gt) ** 2)
#             self.mse.append(mse)

#             psnr = self.psnr_metric(rgb_pred, rgb_gt)
#             self.psnr.append(psnr)

#             ssim = self.ssim_metric(rgb_pred, rgb_gt, batch, epoch)
#             self.ssim.append(ssim)

#     # def summarize(self, epoch=-1):
#     #     if cfg.fast_eval:
#     #         cprint('WARNING: only saving evaluation metrics, no images will be saved!', color='red', attrs=['bold', 'blink'])

#     #     if cfg.dry_run:
#     #         return None

#     #     result_dir = cfg.result_dir
#     #     print(
#     #         colored('the results are saved at {}'.format(result_dir),
#     #                 'yellow'))

#     #     if epoch == -1:
#     #         result_path = os.path.join(cfg.result_dir, 'metrics.npy')
#     #     else:
#     #         result_path = os.path.join(cfg.result_dir, 'metrics_epoch{}.npy'.format(epoch))

#     #     os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
#     #     metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
#     #     np.save(result_path, metrics)

#     #     ret = {}
#     #     print('mse: {}'.format(np.mean(self.mse)))
#     #     print('psnr: {}'.format(np.mean(self.psnr)))
#     #     print('ssim: {}'.format(np.mean(self.ssim)))
#     #     print('lpips: {}'.format(np.mean(self.lpips)))

#     #     ret.update({"psnr": np.mean(self.psnr), "ssim": np.mean(self.ssim), "lpips": np.mean(self.lpips)})

#     #     self.mse = []
#     #     self.psnr = []
#     #     self.ssim = []
#     #     self.lpips = []

#     #     return ret


# # def fill_image(img, batch):
# #     orig_H, orig_W = batch['orig_H'].item(), batch['orig_W'].item()
# #     full_img = np.zeros((orig_H, orig_W, 3))
# #     bbox = batch['crop_bbox'][0].detach().cpu().numpy()
# #     height = bbox[1, 1] - bbox[0, 1]
# #     width = bbox[1, 0] - bbox[0, 0]
# #     full_img[bbox[0, 1]:bbox[1, 1],
# #              bbox[0, 0]:bbox[1, 0]] = img[:height, :width]
# #     return full_img
