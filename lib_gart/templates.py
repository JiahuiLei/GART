# A template only handle the query of the
import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from smplx.smplx import SMPLLayer
from smplx.smplx.lbs import blend_shapes, vertices2joints, batch_rigid_transform
from smal.smal_tpg import SMAL
from voxel_deformer import VoxelDeformer

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from model_utils import get_predefined_human_rest_pose, get_predefined_dog_rest_pose


def get_template(
    mode, init_beta, cano_pose_type, voxel_deformer_res, template_model_path=None
):
    if mode == "human":
        template = SMPLTemplate(
            smpl_model_path=template_model_path,
            init_beta=init_beta,
            cano_pose_type=cano_pose_type,
            voxel_deformer_res=voxel_deformer_res,
        )
    elif mode == "dog":
        template = SMALTemplate(
            init_beta=init_beta,
            cano_pose_type=cano_pose_type,
            voxel_deformer_res=voxel_deformer_res,
        )
    else:
        raise ValueError(f"Unknown mode {mode}")
    return template


class SMPLTemplate(nn.Module):
    def __init__(self, smpl_model_path, init_beta, cano_pose_type, voxel_deformer_res):
        super().__init__()
        self.dim = 24
        self._template_layer = SMPLLayer(model_path=smpl_model_path)

        if init_beta is None:
            init_beta = np.zeros(10)
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)
        self.cano_pose_type = cano_pose_type
        self.name = "smpl"

        can_pose = get_predefined_human_rest_pose(cano_pose_type)
        can_pose = axis_angle_to_matrix(torch.cat([torch.zeros(1, 3), can_pose], 0))
        self.register_buffer("canonical_pose", can_pose)

        init_smpl_output = self._template_layer(
            betas=init_beta[None],
            body_pose=can_pose[None, 1:],
            global_orient=can_pose[None, 0],
            return_full_pose=True,
        )
        J_canonical, A0 = init_smpl_output.J, init_smpl_output.A
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv[0])
        self.register_buffer("J_canonical", J_canonical)

        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        W_init = self._template_layer.lbs_weights  # 6890,24

        self.voxel_deformer = VoxelDeformer(
            vtx=v_init[None],
            vtx_features=W_init[None],
            resolution_dhw=[
                voxel_deformer_res // 4,
                voxel_deformer_res,
                voxel_deformer_res,
            ],
        )

        # * Important, record first joint position, because the global orientation is rotating using this joint position as center, so we can compute the action on later As
        j0_t = init_smpl_output.joints[0, 0]
        self.register_buffer("j0_t", j0_t)
        return

    def get_init_vf(self):
        init_smpl_output = self._template_layer(
            betas=self.init_beta[None],
            body_pose=self.canonical_pose[None, 1:],
            global_orient=self.canonical_pose[None, 0],
            return_full_pose=True,
        )
        v_init = init_smpl_output.vertices  # 1,6890,3
        v_init = v_init[0]
        faces = self._template_layer.faces_tensor
        return v_init, faces

    def get_rot_action(self, axis_angle):
        # apply this action to canonical additional bones
        # axis_angle: B,3
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)  # B,3,3
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)  # B,3,3
        t0 = self.j0_t[None].expand(B, -1)  # B,3
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)  # B,4,4
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T  # B,4,4

    def forward(self, theta=None, xyz_canonical=None):
        # skinning
        if theta is None:
            A = None
        else:
            assert (
                theta.ndim == 3 and theta.shape[-1] == 3
            ), "pose should have shape Bx24x3, in axis-angle format"
            nB = len(theta)
            _, A = batch_rigid_transform(
                axis_angle_to_matrix(theta),
                self.J_canonical.expand(nB, -1, -1),
                self._template_layer.parents,
            )
            A = torch.einsum("bnij, njk->bnik", A, self.A0_inv)  # B,24,4,4

        if xyz_canonical is None:
            # forward theta only
            W = None
        else:
            W = self.voxel_deformer(xyz_canonical)  # B,N,24+K
        return W, A


class SMALTemplate(nn.Module):
    def __init__(self, init_beta, cano_pose_type, voxel_deformer_res):
        super().__init__()
        self.dim = 35
        self._template_layer = SMAL()

        if init_beta is None:
            init_beta = np.zeros(30)
        init_beta = torch.as_tensor(init_beta, dtype=torch.float32).cpu()
        self.register_buffer("init_beta", init_beta)

        self.cano_pose_type = cano_pose_type
        self.name = "smal"
        
        can_pose = get_predefined_dog_rest_pose(cano_pose_type)
        can_pose = torch.cat([torch.zeros(3), can_pose], 0)
        self.register_buffer("canonical_pose", can_pose)

        v_init, (A0, W0, T0, joint_xyz) = self._template_layer(
            beta=self.init_beta[None],
            betas_limbs=can_pose[None, -7:],
            pose=axis_angle_to_matrix(can_pose[None, :-7].reshape(1, 35, 3)),
        )
        A0_inv = torch.inverse(A0)
        self.register_buffer("A0_inv", A0_inv[0])

        self.voxel_deformer = VoxelDeformer(
            vtx=v_init,
            vtx_features=W0,
            resolution_dhw=[
                voxel_deformer_res,
                voxel_deformer_res // 2,
                voxel_deformer_res,
            ],
            short_dim_dhw=1,
            long_dim_dhw=2,
        )

        j0_t = joint_xyz[0, 0]
        self.register_buffer("j0_t", j0_t)
        return

    @torch.no_grad()
    def get_init_vf(self):
        v_init, (A0, W0, T0, _) = self._template_layer(
            beta=self.init_beta[None],
            betas_limbs=self.canonical_pose[None, -7:],
            pose=axis_angle_to_matrix(self.canonical_pose[None, :-7].reshape(1, 35, 3)),
        )
        faces = self._template_layer.faces
        return v_init[0], torch.as_tensor(faces, dtype=torch.long)

    def get_rot_action(self, axis_angle):
        # apply this action to canonical additional bones
        # axis_angle: B,3
        assert axis_angle.ndim == 2 and axis_angle.shape[-1] == 3
        B = len(axis_angle)
        R = axis_angle_to_matrix(axis_angle)  # B,3,3
        I = torch.eye(3).to(R)[None].expand(B, -1, -1)  # B,3,3
        t0 = self.j0_t[None].expand(B, -1)  # B,3
        T = torch.eye(4).to(R)[None].expand(B, -1, -1)  # B,4,4
        T[:, :3, :3] = R
        T[:, :3, 3] = torch.einsum("bij, bj -> bi", I - R, t0)
        return T  # B,4,4

    def forward(self, theta, xyz_canonical):
        if theta is None:
            A = None
        else:
            B = len(theta)
            pose = theta[:, :-7]
            betas_limbs = theta[:, -7:]
            _v, (A, W, T, _) = self._template_layer(
                beta=self.init_beta[None].expand(B, -1),
                betas_limbs=betas_limbs,
                pose=axis_angle_to_matrix(pose.reshape(B, 35, 3)),
            )
            A = torch.einsum("bnij, njk->bnik", A, self.A0_inv)  # B,24,4,4
        if xyz_canonical is None:
            # forward theta only
            W = None
        else:
            W = self.voxel_deformer(xyz_canonical)  # B,N,24
        return W, A


if __name__ == "__main__":
    from transforms3d.euler import euler2mat

    template = get_template(
        "human",
        None,
        "da_pose",
        32,
        template_model_path="../../data/smpl_model/SMPL_NEUTRAL.pkl",
    )

    xyz_canonical = torch.rand(1, 6890, 3)

    pose0 = torch.rand(1, 24, 3)
    pose0[0, 0] = 0.0

    _, A0 = template(pose0, xyz_canonical)
    A0 = A0[0]

    R0 = axis_angle_to_matrix(pose0[0, 0])
    dR = torch.from_numpy(euler2mat(np.pi / 4, np.pi / 4, np.pi / 4, "syxz")).float()
    R1 = dR @ R0

    pose1 = pose0.clone()
    pose1[0, 0] = matrix_to_axis_angle(R1[None])[0]
    _, A1 = template(pose1, xyz_canonical)
    A1 = A1[0]

    action = template.get_rot_action(matrix_to_axis_angle(dR[None]))

    for i in range(len(A0)):
        _A0, _A1 = A0[i], A1[i]
        _A0_inv = torch.inverse(_A0)
        dA = _A1 @ _A0_inv
        print(abs(dA - action).max())
        # print(dA)

    print()
