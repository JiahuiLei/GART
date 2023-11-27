import torch
import matplotlib.pyplot as plt
import numpy as np


def viz_render(gt_rgb, gt_mask, pred_pkg, save_path=None):
    pred_rgb = pred_pkg["rgb"].permute(1, 2, 0)
    pred_mask = pred_pkg["alpha"].squeeze(0)
    pred_depth = pred_pkg["dep"].squeeze(0)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(torch.clamp(gt_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("GT"), plt.axis("off")
    plt.subplot(1, 5, 2)
    plt.imshow(torch.clamp(pred_rgb, 0.0, 1.0).detach().cpu().numpy())
    plt.title("Pred view"), plt.axis("off")
    plt.subplot(1, 5, 3)
    error = torch.clamp(abs(pred_rgb - gt_rgb), 0.0, 1.0).detach().cpu().numpy().max(axis=-1)
    cmap = plt.imshow(error)
    plt.title("Render Error (max in rgb)"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.subplot(1, 5, 4)
    error = torch.clamp(pred_mask - gt_mask, -1.0, 1.0).detach().cpu().numpy()
    cmap = plt.imshow(error)
    plt.title("(Pr - GT) Mask Error"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)
    
    plt.subplot(1, 5, 5)
    depth = pred_depth.detach().cpu().numpy()
    cmap = plt.imshow(depth)
    plt.title("Pred Depth"), plt.axis("off")
    plt.colorbar(cmap, shrink=0.8)

    plt.tight_layout()
    fig.canvas.draw()
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)
    return fig_np
