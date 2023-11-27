import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F

from optim_utils import *
from init_helpers import get_near_mesh_init_geo_values, get_on_mesh_init_geo_values, get_inside_mesh_init_geo_values
import logging

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from model_utils import sph_order2nfeat
from pytorch3d.ops import knn_points


class AdditionalBones(nn.Module):
    def __init__(
        self,  # additional bones
        num_bones: int = 0,
        total_t: int = 0,  # any usage of time should use this!
        mode="pose-mlp",
        # pose-mlp
        pose_dim=23 * 3,
        mlp_hidden_dims=[256, 256, 256, 256],
        mlp_act=nn.LeakyReLU,
        # pose+t-mlp
    ):
        super().__init__()
        self.num_bones = num_bones
        if self.num_bones == 0:
            return
        self.mode = mode
        assert self.mode in ["pose-mlp", "pose+t-mlp", "delta-list", "list"]
        self.total_t = total_t

        if self.mode == "pose-mlp":
            self.pose_dim = pose_dim
            self.mlp_layers = nn.ModuleList()
            c_in = self.pose_dim
            for c_out in mlp_hidden_dims:
                self.mlp_layers.append(nn.Sequential(nn.Linear(c_in, c_out), mlp_act()))
                c_in = c_out
            self.mlp_output_head = nn.Linear(c_in, 7 * self.num_bones, bias=False)
            with torch.no_grad():
                self.mlp_output_head.weight.data = (
                    torch.randn_like(self.mlp_output_head.weight.data) * 1e-3
                )
        elif self.mode == "delta-list":
            self.dr_list = nn.Parameter(torch.zeros(self.total_t, num_bones, 3))
            self.dt_list = nn.Parameter(torch.zeros(self.total_t, num_bones, 3))
        else:
            raise NotImplementedError()

        return

    def forward(self, pose=None, t=None, As=None):
        if self.num_bones == 0:
            # * No additional bones
            return None
        if As is not None:
            # * Directly return if As already provided
            return As
        if self.mode == "pose-mlp":
            assert pose is not None
            assert pose.ndim == 2 and pose.shape[1] == self.pose_dim
            B = len(pose)
            x = pose
            for layer in self.mlp_layers:
                x = layer(x)
            x = self.mlp_output_head(x).reshape(B, -1, 7)
            q, t = x[:, :, :4], x[:, :, 4:]
            q[..., 0] = q[..., 0] + 1.0
            q = F.normalize(q, dim=-1)
            R = quaternion_to_matrix(q)
            Rt = torch.cat([R, t[:, :, :, None]], dim=-1)
            bottom = torch.zeros_like(Rt[:, :, 0:1])
            bottom[:, :, :, -1] = 1.0
            As = torch.cat([Rt, bottom], dim=2)
            return As
        elif self.mode == "delta-list":
            As = self._roll_out_continuous_T()
            if t is None:
                B = len(pose)
                # # ! If no time is set, now return eye(4)
                # ret = (
                #     torch.eye(4)
                #     .to(As.device)[None, None]
                #     .repeat(B, self.num_bones, 1, 1)
                # )
                # ! If no time is set, now return first frame
                ret = As[0][None].repeat(B, 1, 1, 1)
            else:
                if isinstance(t, int):
                    t = torch.tensor([t]).to(As.device)
                ret = As[t]
            return ret
        else:
            raise NotImplementedError()

        return  # As in canonical frame

    def _roll_out_continuous_T(self):
        # ! this assumes continuous frames, single frame!
        R = axis_angle_to_matrix(self.dr_list)
        dT = (
            torch.eye(4).to(R.device)[None, None].repeat(self.total_t, R.shape[1], 1, 1)
        )
        dT[:, :, :3, :3] = dT[:, :, :3, :3] * 0 + R
        dT[:, :, :3, 3] = dT[:, :, :3, 3] * 0 + self.dt_list
        T = [dT[0]]
        for i in range(1, self.total_t):
            T.append(torch.einsum("nij, njk->nik", T[-1], dT[i]))
        T = torch.stack(T, dim=0)
        return T


class GaussianTemplateModel(nn.Module):
    def __init__(
        self,
        template,
        add_bones: AdditionalBones,
        ##################################
        # attr config
        w_correction_flag=True,
        # w_rest_dim=0,  # additional skinnign weight
        f_localcode_dim=0,
        max_sph_order=0,
        w_memory_type="point",
        ##################################
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0,
        # geo init
        init_mode="on_mesh",
        opacity_init_value=0.9,  # the init value of opacity
        # on mesh init params
        onmesh_init_subdivide_num=0,
        onmesh_init_scale_factor=1.0,
        onmesh_init_thickness_factor=0.5,
        # near mesh init params
        scale_init_value=0.01,  # the init value of scale
        nearmesh_init_num=10000,
        nearmesh_init_std=0.1,
        ##################################
    ) -> None:
        super().__init__()

        self.template = template
        self.num_bones = template.voxel_deformer.num_bones
        self.add_bones = add_bones
        self.num_add_bones = add_bones.num_bones

        self.max_scale = max_scale
        self.min_scale = min_scale
        self._init_act(self.max_scale, self.min_scale)
        self.opacity_init_logit = self.o_inv_act(opacity_init_value)

        # * init geometry
        if init_mode == "on_mesh":
            x, q, s, o = get_on_mesh_init_geo_values(
                template,
                on_mesh_subdivide=onmesh_init_subdivide_num,
                scale_init_factor=onmesh_init_scale_factor,
                thickness_init_factor=onmesh_init_thickness_factor,
                max_scale=max_scale,
                min_scale=min_scale,
                s_inv_act=self.s_inv_act,
                opacity_init_logit=self.opacity_init_logit,
            )
        elif init_mode == "near_mesh":
            self.scale_init_logit = self.s_inv_act(scale_init_value)
            x, q, s, o = get_near_mesh_init_geo_values(
                template,
                scale_base_logit=self.scale_init_logit,
                opacity_base_logit=self.opacity_init_logit,
                random_init_num=nearmesh_init_num,
                random_init_std=nearmesh_init_std,
            )
        elif init_mode == "in_mesh":
            self.scale_init_logit = self.s_inv_act(scale_init_value)
            x, q, s, o = get_inside_mesh_init_geo_values(
                template,
                scale_base_logit=self.scale_init_logit,
                opacity_base_logit=self.opacity_init_logit,
                random_init_num=nearmesh_init_num,
            )
        else:
            raise NotImplementedError(f"Unknown init_mode {init_mode}")
        self._xyz = nn.Parameter(x)
        self._rotation = nn.Parameter(q)
        self._scaling = nn.Parameter(s)
        self._opacity = nn.Parameter(o)

        # * init attributes
        self.w_memory_type = w_memory_type
        assert self.w_memory_type in ["point", "voxel"], f"Unknown {w_memory_type}"

        self.max_sph_order = max_sph_order
        self.w_dc_dim = self.template.dim if w_correction_flag else 0
        self.w_rest_dim = self.add_bones.num_bones
        self.f_localcode_dim = f_localcode_dim

        sph_rest_dim = 3 * (sph_order2nfeat(self.max_sph_order) - 1)
        self._features_dc = nn.Parameter(torch.zeros_like(self._xyz))
        self._features_rest = nn.Parameter(torch.zeros(self.N, sph_rest_dim))

        # * Different implementation of smoothness
        if self.w_memory_type == "point":
            self._w_correction_dc = nn.Parameter(torch.zeros(self.N, self.w_dc_dim))
            self._w_correction_rest = nn.Parameter(
                torch.ones(self.N, self.w_rest_dim) * 1e-4
            )
        elif self.w_memory_type == "voxel":
            self._w_correction_dc = nn.Parameter(torch.zeros(self.N, 0))
            self._w_correction_rest = nn.Parameter(torch.zeros(self.N, 0))
            if self.w_dc_dim > 0:
                self.template.voxel_deformer.enable_voxel_correction()
            if self.w_rest_dim > 0:
                self.template.voxel_deformer.enable_additional_correction(
                    self.w_rest_dim
                )
        elif self.w_memory_type == "hash":
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError(f"Unknown {w_memory_type}")

        self._features_localcode = nn.Parameter(
            torch.zeros(self.N, self.f_localcode_dim)
        )

        assert self.f_localcode_dim == 0, "TODO, add local mlp ablation"

        # * States
        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        self.op_update_exclude = ["add_bones"]
        if self.w_memory_type != "point":
            self.op_update_exclude.extend(["w_dc_vox", "w_rest_vox"])
        # self.summary()
        return

    def summary(self):
        # logging.info number of parameters per pytorch sub module
        msg = ""
        for name, param in self.named_parameters():
            if name.startswith("add_bones"):
                continue # compact print
            msg = msg + f"[{name}:{param.numel()/1e3:.1f}K] " 
            # logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info(msg)
        return

    def _init_act(self, max_s_value, min_s_value):
        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.logit(y)
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y

        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act

        return

    @property
    def N(self):
        return len(self._xyz)

    @property
    def get_x(self):
        return self._xyz

    @property
    def get_R(self):
        return quaternion_to_matrix(self._rotation)

    @property
    def get_o(self):
        return self.o_act(self._opacity)

    @property
    def get_s(self):
        return self.s_act(self._scaling)

    @property
    def get_c(self):
        return torch.cat([self._features_dc, self._features_rest], dim=-1)

    def cache_for_fast(self):
        _cached_W, _ = self.template.forward(None, self._xyz)
        self._cached_W = _cached_W.detach().clone()
        return

    def forward(
        self, theta, trans, additional_dict={}, active_sph_order=None, fast=False
    ):
        # * fast will use the cached per point attr, no query anymore
        # TODO: the additional dict contain info to do flexible skinning: it can contain the As directly for optimization, or it can contain t index to query some buffers to provide As, or it can contain t along with the input theta to query some MLP;

        # TODO: if use vol memory, every forward update self.xxx, and remove them from parameters, pretend that the attributes are per point, but actually they are queried every forward

        # theta: B,24,3; trans: B,3
        B = len(theta)
        if active_sph_order is None:
            active_sph_order = self.max_sph_order
        else:
            assert (
                active_sph_order <= self.max_sph_order
            ), "active_sph_order should be smaller"
        sph_dim = 3 * sph_order2nfeat(active_sph_order)

        xyz = self.get_x
        mu_can = xyz
        frame_can = self.get_R
        s = self.get_s
        o = self.get_o
        sph = self.get_c[:, :sph_dim]

        mu_can = mu_can[None].expand(B, -1, -1)
        frame_can = frame_can[None].expand(B, -1, -1, -1)

        if fast:
            # only forward skeleton, no query voxel
            _, A = self.template.forward(theta, None)
            W = self._cached_W[None].expand(B, -1, -1)
        else:
            W, A = self.template.forward(theta, mu_can)
        if self._w_correction_dc.shape[-1] > 0:
            W = W + self._w_correction_dc[None]
        T = torch.einsum("bnj, bjrc -> bnrc", W[..., : self.num_bones], A)

        # * additional correction here
        if "pose" not in additional_dict.keys():
            # maybe later we want to viz the different pose effect in cano
            additional_dict["pose"] = theta.reshape(B, -1)[:, 3:]
        add_A = self.add_bones(**additional_dict)
        if add_A is not None:
            if theta.ndim == 2:
                global_axis_angle = theta[:, :3]
            else:
                global_axis_angle = theta[:, 0]
            global_orient_action = self.template.get_rot_action(global_axis_angle)  # B,4,4
            add_A = torch.einsum("bij, bnjk -> bnik", global_orient_action, add_A)

            if self.w_memory_type == "point":
                assert self._w_correction_rest.shape[-1] > 0
                add_W = self._w_correction_rest[None].expand(B, -1, -1)
            elif self.w_memory_type == "voxel":
                add_W = W[..., self.num_bones :]

            add_T = torch.einsum("bnj, bjrc -> bnrc", add_W, add_A)
            T = T + add_T  # Linear
            additional_dict["As"] = add_A

        R, t = T[:, :, :3, :3], T[:, :, :3, 3]  # B,N,3,3; B,N,3

        mu = torch.einsum("bnij,bnj->bni", R, mu_can) + t  # B,N,3
        frame = torch.einsum("bnij,bnjk->bnik", R, frame_can)  # B,N,3,3

        s = s[None].expand(B, -1, -1)  # B,N,1
        o = o[None].expand(B, -1, -1)  # B,N,1
        sph = sph[:, :sph_dim][None].expand(B, -1, -1)  # B,N,C

        mu = mu + trans[:, None, :]

        return mu, frame, s, o, sph, additional_dict

    def compute_reg(self, K):
        # !can cancel the knn, but the w reg is critical
        if K > 0:
            xyz = self._xyz
            # todo: this can be cached and updated every several steps!!
            dist_sq, nn_ind, _ = knn_points(xyz[None], xyz[None], K=K, return_nn=False)
            nn_ind = nn_ind.squeeze(0)
            # reg the std inside knn
            q = self._rotation[nn_ind, :]  # N,K,4
            s = self.get_s[nn_ind, :]  # N,K,3
            o = self.get_o[nn_ind, :]  # N,K,1
            q_std = q.std(dim=1).mean()
            s_std = s.std(dim=1).mean()
            o_std = o.std(dim=1).mean()

            cd = self._features_dc[nn_ind, :]  # N,K,3
            ch = self._features_rest[nn_ind, :]  # N,K,C
            cd_std = cd.std(dim=1).mean()
            ch_std = ch.std(dim=1).mean()
            if ch.shape[-1] == 0:
                ch_std = torch.zeros_like(ch_std)

            w = self._w_correction_dc[nn_ind, :]  # N,K,3
            w_rest = self._w_correction_rest[nn_ind, :]  # N,K,C
            f = self._features_localcode[nn_ind, :]  # N,K,C
            w_std = w.std(dim=1).mean()
            w_rest_std = w_rest.std(dim=1).mean()
            f_std = f.std(dim=1).mean()
            if w.shape[-1] == 0:
                w_std = torch.zeros_like(cd_std)
            if w_rest.shape[-1] == 0:
                w_rest_std = torch.zeros_like(cd_std)
            if f.shape[-1] == 0:
                f_std = torch.zeros_like(cd_std)
        else:
            dummy = torch.zeros(1).to(self._xyz).squeeze()
            q_std, s_std, o_std = dummy, dummy, dummy
            cd_std, ch_std = dummy, dummy
            w_std, w_rest_std, f_std = dummy, dummy, dummy
            dist_sq = dummy

        w_norm = self._w_correction_dc.norm(dim=-1).mean()  # N
        w_rest_norm = self._w_correction_rest.norm(dim=-1).mean()  # N

        if self.w_memory_type == "voxel":
            # update the w related std and norm
            w_std = self.template.voxel_deformer.get_tv("dc")
            w_rest_std = self.template.voxel_deformer.get_tv("rest")
            w_norm = self.template.voxel_deformer.get_mag("dc")
            w_rest_norm = self.template.voxel_deformer.get_mag("rest")

        max_s_square = torch.mean((self.get_s.max(dim=1).values) ** 2)

        return (
            q_std,
            s_std,
            o_std,
            cd_std,
            ch_std,
            w_std,
            w_rest_std,
            f_std,
            w_norm,
            w_rest_norm,
            dist_sq.mean(),
            max_s_square,
        )

    def get_optimizable_list(
        self,
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
        lr_w=0.001,
        lr_w_rest=0.001,
        lr_f=0.0001,
    ):
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._rotation], "lr": lr_q, "name": "rotation"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
            {"params": [self._features_rest], "lr": lr_sph_rest, "name": "f_rest"},
            {"params": [self._w_correction_dc], "lr": lr_w, "name": "w_dc"},
            {"params": [self._w_correction_rest], "lr": lr_w_rest, "name": "w_rest"},
            {"params": [self._features_localcode], "lr": lr_f, "name": "f_localcode"},
        ]
        if self.w_memory_type == "voxel":
            if self.w_dc_dim > 0:
                l.append(
                    {
                        "params": [self.template.voxel_deformer.voxel_w_correction],
                        "lr": lr_w,
                        "name": "w_dc_vox",
                    }
                )
            if self.w_rest_dim > 0:
                l.append(
                    {
                        "params": [self.template.voxel_deformer.additional_correction],
                        "lr": lr_w_rest,
                        "name": "w_rest_vox",
                    }
                )
        return l

    # * Gaussian Control
    def record_xyz_grad_radii(self, viewspace_point_tensor, radii, update_filter):
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor) == self.N
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.xyz_gradient_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s,
        new_o,
        new_sph_dc,
        new_sph_rest,
        new_w_dc,
        new_w_rest,
        new_localcode,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "f_rest": new_sph_rest,
            "opacity": new_o,
            "scaling": new_s,
            "rotation": new_r,
            "w_dc": new_w_dc,
            "w_rest": new_w_rest,
            "f_localcode": new_localcode,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # First cat to optimizer and then return to self
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._w_correction_dc = optimizable_tensors["w_dc"]
        self._w_correction_rest = optimizable_tensors["w_rest"]
        self._features_localcode = optimizable_tensors["f_localcode"]

        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], device="cuda")
        self.xyz_gradient_denom = torch.zeros(self._xyz.shape[0], device="cuda")
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )
        return

    def _densify_and_clone(self, optimizer, grad_norm, grad_threshold, scale_th):
        # Extract points that satisfy the gradient condition
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device="cuda")
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_s, dim=1).values <= scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_w_dc = self._w_correction_dc[selected_pts_mask]
        new_w_rest = self._w_correction_rest[selected_pts_mask]
        new_localcode = self._features_localcode[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )

        return len(new_xyz)

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        # Extract points that satisfy the gradient condition
        _scaling = self.get_s
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device="cuda")
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(_scaling, dim=1).values > scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        stds = _scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        new_w_dc = self._w_correction_dc[selected_pts_mask].repeat(N, 1)
        new_w_rest = self._w_correction_rest[selected_pts_mask].repeat(N, 1)
        new_localcode = self._features_localcode[selected_pts_mask].repeat(N, 1)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return len(new_xyz)

    def densify(self, optimizer, max_grad, percent_dense, extent, verbose=True):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0

        # n_clone = self._densify_and_clone(optimizer, grads, max_grad)
        n_clone = self._densify_and_clone(
            optimizer, grads, max_grad, percent_dense * extent
        )
        n_split = self._densify_and_split(
            optimizer, grads, max_grad, percent_dense * extent, N=2
        )

        if verbose:
            logging.info(f"Densify: Clone[+] {n_clone}, Split[+] {n_split}")
            # logging.info(f"Densify: Clone[+] {n_clone}")
        # torch.cuda.empty_cache()
        return

    def random_grow(self, optimizer, num_factor=0.05, std=0.1, init_opa_value=0.1):
        # * New operation, randomly add largely disturbed points to the geometry
        ind = torch.randperm(self.N)[: int(self.N * num_factor)]
        selected_pts_mask = torch.zeros(self.N, dtype=bool, device="cuda")
        selected_pts_mask[ind] = True

        new_xyz = self._xyz[selected_pts_mask]
        noise = torch.randn_like(new_xyz) * std
        new_xyz = new_xyz + noise
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        new_opacities = torch.ones_like(self._opacity[selected_pts_mask])
        new_opacities = new_opacities * self.o_inv_act(init_opa_value)

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_w_dc = self._w_correction_dc[selected_pts_mask]
        new_w_rest = self._w_correction_rest[selected_pts_mask]
        new_localcode = self._features_localcode[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rotation,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            new_sph_rest=new_features_rest,
            new_w_dc=new_w_dc,
            new_w_rest=new_w_rest,
            new_localcode=new_localcode,
        )
        logging.info(f"Random grow: {len(new_xyz)}")
        return len(new_xyz)

    def prune_points(self, optimizer, min_opacity, max_screen_size, verbose=True):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        if getattr(self, "color_memory", None) is None:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._w_correction_dc = optimizable_tensors["w_dc"]
        self._w_correction_rest = optimizable_tensors["w_rest"]
        self._features_localcode = optimizable_tensors["f_localcode"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # torch.cuda.empty_cache()
        return

    @torch.no_grad()
    def regaussian(self, optimizer, max_scale=0.03):
        # raise NotImplementedError("TODO, like split")
        # * New operation, manually split the large gaussians with smaller ones to approximate
        # * Now, try bi-split

        # Extract points that satisfy the gradient condition
        _scaling = self.get_s
        selected_pts_mask = torch.max(_scaling, dim=1).values > max_scale

        step = 0
        before_num = self.N
        while selected_pts_mask.any():
            # This can be done more than 3 times, becuase there may be huge gaussians, which should be devided several times
            fg_xyz = self._xyz[selected_pts_mask]
            fg_scale = _scaling[selected_pts_mask]
            fg_frame = quaternion_to_matrix(self._rotation[selected_pts_mask])
            # each column is the direction of axis in global frame
            axis_ind = torch.argmax(fg_scale, dim=1)
            axis_scale = fg_scale.max(dim=1).values
            # select column
            axis_dir = torch.gather(
                fg_frame, dim=2, index=axis_ind[:, None, None].expand(-1, 3, -1)
            ).squeeze(
                -1
            )  # N,3
            new_x1 = fg_xyz + axis_dir.squeeze() * axis_scale[:, None] / 2.0
            new_x2 = fg_xyz - axis_dir.squeeze() * axis_scale[:, None] / 2.0
            # Repeat will change [1,2,3...] to [1,2,3..., 1,2,3...]
            new_xyz = torch.cat([new_x1, new_x2], dim=0).reshape(-1, 3)
            new_scaling = _scaling[selected_pts_mask]
            new_scaling = torch.scatter(
                new_scaling,
                dim=1,
                index=axis_ind[:, None],
                src=axis_scale[:, None] / 2.0,
            ).repeat(2, 1)
            new_scaling = torch.clamp(
                new_scaling, max=self.max_scale, min=self.min_scale
            )
            new_scaling = self.s_inv_act(new_scaling)
            new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)
            new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1)
            new_opacities = self._opacity[selected_pts_mask].repeat(2, 1)
            new_w_dc = self._w_correction_dc[selected_pts_mask].repeat(2, 1)
            new_w_rest = self._w_correction_rest[selected_pts_mask].repeat(2, 1)
            new_localcode = self._features_localcode[selected_pts_mask].repeat(2, 1)

            self._densification_postprocess(
                optimizer,
                new_xyz=new_xyz.float(),
                new_r=new_rotation.float(),
                new_s=new_scaling.float(),
                new_o=new_opacities.float(),
                new_sph_dc=new_features_dc.float(),
                new_sph_rest=new_features_rest.float(),
                new_w_dc=new_w_dc.float(),
                new_w_rest=new_w_rest.float(),
                new_localcode=new_localcode.float(),
            )

            prune_filter = torch.cat(
                (
                    selected_pts_mask,
                    torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool),
                )
            )
            self._prune_points(optimizer, prune_filter)

            step += 1
            logging.info(
                f"Regaussian-[{step}], {selected_pts_mask.sum()} ({selected_pts_mask.float().mean()*100}% pts-scale>{max_scale})"
            )

            _scaling = self.get_s
            selected_pts_mask = torch.max(_scaling, dim=1).values > max_scale
        logging.info(f"Re-gaussian: {before_num} -> {self.N}")
        return

    def reset_opacity(self, optimizer, value=0.01, verbose=True):
        opacities_new = self.o_inv_act(
            torch.min(self.o_act(self._opacity), torch.ones_like(self._opacity) * value)
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, opacities_new, "opacity"
        )
        if verbose:
            logging.info(f"Reset opacity to {value}")
        self._opacity = optimizable_tensors["opacity"]

    def load(self, ckpt):
        # because N changed, have to re-init the buffers
        self._xyz = nn.Parameter(torch.as_tensor(ckpt["_xyz"], dtype=torch.float32))

        self._features_dc = nn.Parameter(
            torch.as_tensor(ckpt["_features_dc"], dtype=torch.float32)
        )
        self._features_rest = nn.Parameter(
            torch.as_tensor(ckpt["_features_rest"], dtype=torch.float32)
        )
        self._opacity = nn.Parameter(
            torch.as_tensor(ckpt["_opacity"], dtype=torch.float32)
        )
        self._scaling = nn.Parameter(
            torch.as_tensor(ckpt["_scaling"], dtype=torch.float32)
        )
        self._rotation = nn.Parameter(
            torch.as_tensor(ckpt["_rotation"], dtype=torch.float32)
        )
        self._w_correction_dc = nn.Parameter(
            torch.as_tensor(ckpt["_w_correction_dc"], dtype=torch.float32)
        )
        self._w_correction_rest = nn.Parameter(
            torch.as_tensor(ckpt["_w_correction_rest"], dtype=torch.float32)
        )
        self._features_localcode = nn.Parameter(
            torch.as_tensor(ckpt["_features_localcode"], dtype=torch.float32)
        )
        self.xyz_gradient_accum = torch.as_tensor(
            ckpt["xyz_gradient_accum"], dtype=torch.float32
        )
        self.xyz_gradient_denom = torch.as_tensor(
            ckpt["xyz_gradient_denom"], dtype=torch.int64
        )
        self.max_radii2D = torch.as_tensor(ckpt["max_radii2D"], dtype=torch.float32)

        # * add bones may have different total_t
        if "add_bones.dt_list" in ckpt.keys():
            self.add_bones.total_t = ckpt["add_bones.dt_list"].shape[0]
            self.add_bones.dt_list = nn.Parameter(
                torch.as_tensor(ckpt["add_bones.dt_list"], dtype=torch.float32)
            )
            self.add_bones.dr_list = nn.Parameter(
                torch.as_tensor(ckpt["add_bones.dr_list"], dtype=torch.float32)
            )
        # load others
        self.load_state_dict(ckpt, strict=True)
        # this is critical, reinit the funcs
        self._init_act(self.max_scale, self.min_scale)
        return


if __name__ == "__main__":
    import os.path as osp
    from templates import get_template

    # model = GaussianTemplateModelGridX(mode="dog").cuda()
    # theta = torch.zeros(2, 35 * 3 + 7).cuda()
    # trans = torch.zeros(2, 3).cuda()

    template = get_template(
        mode="human",
        template_model_path="../../data/smpl_model/SMPL_NEUTRAL.pkl",
        init_beta=None,
        cano_pose_type="t_pose",
        voxel_deformer_res=64,
    )

    add_bones = AdditionalBones(
        num_bones=16,
        total_t=100,  # any usage of time should use this!
        mode="pose-mlp",
    )

    model = GaussianTemplateModel(
        template=template, add_bones=add_bones, w_correction_flag=True
    ).cuda()

    theta = torch.zeros(2, 24, 3).cuda()
    trans = torch.zeros(2, 3).cuda()

    ret = model(theta, trans, {"t": 0})
    print(ret)
    print("Done")
