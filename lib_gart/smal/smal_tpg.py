"""
PyTorch implementation of the SMAL/SMPL model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import chumpy as ch
import os.path
from torch import nn
from torch.autograd import Variable
import pickle as pkl

from .SMAL_configs import SMAL_MODEL_CONFIG
from .batch_lbs import (
    batch_global_rigid_transformation_biggs,
    get_bone_length_scales,
    get_beta_scale_mask,
)


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


# class SMAL(object):
class SMAL(nn.Module):
    def __init__(self, use_smal_betas=True):
        super(SMAL, self).__init__()

        smal_model_type = "39dogs_norm_newv3"
        pkl_path = SMAL_MODEL_CONFIG[smal_model_type]["smal_model_path"]

        self.logscale_part_list = SMAL_MODEL_CONFIG[smal_model_type]["logscale_part_list"]
        self.betas_scale_mask = get_beta_scale_mask(part_list=self.logscale_part_list)
        self.num_betas_logscale = len(self.logscale_part_list)
        self.use_smal_betas = use_smal_betas

        # -- Load SMPL params --
        try:
            with open(pkl_path, "r") as f:
                dd = pkl.load(f)
        except (UnicodeDecodeError, TypeError) as e:
            with open(pkl_path, "rb") as file:
                u = pkl._Unpickler(file)
                u.encoding = "latin1"
                dd = u.load()

        self.f = dd["f"]
        self.register_buffer("faces", torch.from_numpy(self.f.astype(int)))

        # get the correct template (mean shape)
        v_template = dd["v_template"]
        v = v_template
        self.register_buffer("v_template", torch.Tensor(v))

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd["shapedirs"].shape[-1]

        # Shape blend shape basis
        shapedir = np.reshape(undo_chumpy(dd["shapedirs"]), [-1, self.num_betas]).T
        shapedir.flags["WRITEABLE"] = True  # not sure why this is necessary
        self.register_buffer("shapedirs", torch.Tensor(shapedir))

        # Regressor for joint locations given shape
        self.register_buffer("J_regressor", torch.Tensor(dd["J_regressor"].T.todense()))

        # Pose blend shape basis
        num_pose_basis = dd["posedirs"].shape[-1]

        posedirs = np.reshape(undo_chumpy(dd["posedirs"]), [-1, num_pose_basis]).T
        self.register_buffer("posedirs", torch.Tensor(posedirs))

        # indices of parents for each joints
        self.parents = dd["kintree_table"][0].astype(np.int32)

        # LBS weights
        self.register_buffer("weights", torch.Tensor(undo_chumpy(dd["weights"])))

    def forward(self, beta, betas_limbs, pose=None, trans=None):
        device = beta.device

        betas_logscale = betas_limbs
        nBetas = beta.shape[1]
        num_batch = beta.shape[0]

        # 1. PCA blend shapes
        shapedirs_sel = self.shapedirs[:nBetas, :]
        v_shaped = self.v_template + torch.reshape(
            torch.matmul(beta, shapedirs_sel), [-1, self.size[0], self.size[1]]
        )

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose-dependent deformation: (B, 35, 3, 3)
        Rs = pose
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(device=device), [-1, 306])
        v_posed = (
            torch.reshape(
                torch.matmul(pose_feature, self.posedirs), [-1, self.size[0], self.size[1]]
            )
            + v_shaped
        )

        # 4. limb scaling: it's kind of like streching, think of it as if having a non rigid joint
        # Specifically for animals.
        betas_scale = torch.exp(betas_logscale @ self.betas_scale_mask.to(betas_logscale.device))
        scaling_factors = betas_scale.reshape(-1, 35, 3)
        scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

        # ---------------------------------------------------------------------
        # 5 Forward kinematics & transforms
        joint_location, A = batch_global_rigid_transformation_biggs(
            Rs, J, self.parents, scale_factors_3x3, betas_logscale=betas_logscale
        )

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 35])  # (B, 3889 vertices, 35 joints])

        T = torch.reshape(
            torch.matmul(W, torch.reshape(A, [num_batch, 35, 16])), [num_batch, -1, 4, 4]
        )
        # ---------------------------------------------------------------------

        # 6. Do skinning.
        v_posed_homo = torch.cat(
            [v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=device)], 2
        )
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch, 3)).to(device=device)

        verts = verts + trans[:, None, :]

        return verts, (A, W, T, joint_location)
