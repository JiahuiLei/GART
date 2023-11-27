from matplotlib import pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle
import imageio
import torch
from tqdm import tqdm
import numpy as np
import os, os.path as osp, shutil, sys
from transforms3d.euler import euler2mat
from omegaconf import OmegaConf

from lib_data.get_data import prepare_real_seq
from lib_data.data_provider import DatabasePoseProvider

from lib_gart.templates import get_template
from lib_gart.model import GaussianTemplateModel, AdditionalBones

from lib_gart.optim_utils import *
from lib_render.gauspl_renderer import render_cam_pcl

# from lib_marchingcubes.gaumesh_utils import MeshExtractor
from lib_gart.model_utils import transform_mu_frame

from utils.misc import *
from utils.viz import viz_render

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
import time

from lib_guidance.camera_sampling import sample_camera, fov2K, opencv2blender
import logging

from viz_utils import viz_spinning, viz_human_all, viz_dog_all
from utils.ssim import ssim

import argparse
from datetime import datetime
from test_utils import test

try:
    # from lib_guidance.sd_utils import StableDiffusion
    from lib_guidance.mvdream.mvdream_guidance import MVDream
except:
    logging.warning("No guidance module")


class TGFitter:
    def __init__(
        self,
        log_dir,
        profile_fn,
        mode,
        template_model_path="data/smpl_model/SMPL_NEUTRAL.pkl",
        device=torch.device("cuda:0"),
        **kwargs,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.profile_fn = profile_fn
        try:
            shutil.copy(profile_fn, osp.join(self.log_dir, osp.basename(profile_fn)))
        except:
            pass

        self.mode = mode
        assert self.mode in ["human", "dog"], "Only support human and dog for now"

        self.template_model_path = template_model_path
        self.device = device

        # * auto set attr
        cfg = OmegaConf.load(profile_fn)
        # assign the cfg to self attribute
        for k, v in cfg.items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            setattr(self, k, v)

        # * explicitly set flags
        self.FAST_TRAINING = getattr(self, "FAST_TRAINING", False)

        self.LAMBDA_SSIM = getattr(self, "LAMBDA_SSIM", 0.0)
        self.LAMBDA_LPIPS = getattr(self, "LAMBDA_LPIPS", 0.0)
        if self.LAMBDA_LPIPS > 0:
            from utils.lpips import LPIPS

            self.lpips = LPIPS(net="vgg").to(self.device)
            for param in self.lpips.parameters():
                param.requires_grad = False

        if isinstance(self.RESET_OPACITY_STEPS, int):
            self.RESET_OPACITY_STEPS = [
                i
                for i in range(1, self.TOTAL_steps)
                if i % self.RESET_OPACITY_STEPS == 0
            ]
        if isinstance(self.REGAUSSIAN_STEPS, int):
            self.REGAUSSIAN_STEPS = [
                i for i in range(1, self.TOTAL_steps) if i % self.REGAUSSIAN_STEPS == 0
            ]

        # prepare base R
        if self.mode == "human":
            viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
        else:
            viz_base_R_opencv = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
        viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
        self.viz_base_R = viz_base_R_opencv.to(self.device)

        if self.mode == "human":
            self.reg_base_R_global = (
                matrix_to_axis_angle(
                    torch.as_tensor(euler2mat(np.pi / 2.0, 0, np.pi / 2.0, "sxyz"))[
                        None
                    ]
                )[0]
                .float()
                .to(self.device)
            )
        else:
            # TODO, for generation of dog
            pass

        self.writer = create_log(
            self.log_dir, name=osp.basename(self.profile_fn).split(".")[0], debug=False
        )
        return

    def prepare_fake_data(self, mode, *args, **kwargs):
        if mode == "amass":
            # todo: change to amass
            provider = DatabasePoseProvider(*args, **kwargs, device=torch.device("cpu"))
            return provider
        return provider

    def prepare_real_seq(
        self,
        seq_name,
        dataset_mode,
        split,
        ins_avt_wild_start_end_skip=None,
        image_zoom_ratio=0.5,
        data_stay_gpu_flag=True,
    ):
        provider, dataset = prepare_real_seq(
            seq_name=seq_name,
            dataset_mode=dataset_mode,
            split=split,
            ins_avt_wild_start_end_skip=ins_avt_wild_start_end_skip,
            image_zoom_ratio=getattr(
                self, "IMAGE_ZOOM_RATIO", image_zoom_ratio
            ),  # ! this overwrite the func arg
            balance=getattr(self, "VIEW_BALANCE_FLAG", False),
        )
        provider.to(self.device)
        if getattr(self, "DATA_STAY_GPU_FLAG", data_stay_gpu_flag):
            provider.move_images_to_device(self.device)
        provider.viz_selection_prob(
            osp.join(self.log_dir, f"split_{split}_view_prob.png")
        )
        return provider, dataset

    def load_saved_model(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = osp.join(self.log_dir, "model.pth")
        ret = self._get_model_optimizer(betas=None)
        model = ret[0]

        model.load(torch.load(ckpt_path))
        model.to(self.device)
        model.eval()
        logging.info("After loading:")
        model.summary()
        return model

    def _get_model_optimizer(self, betas, add_bones_total_t=0):
        seed_everything(self.SEED)

        template = get_template(
            mode=self.mode,
            template_model_path=self.template_model_path,
            init_beta=betas,
            cano_pose_type=getattr(self, "CANO_POSE_TYPE", "t_pose"),
            voxel_deformer_res=getattr(self, "VOXEL_DEFORMER_RES", 64),
        )

        add_bones = AdditionalBones(
            num_bones=getattr(self, "W_REST_DIM", 0),
            mode=getattr(self, "W_REST_MODE", "pose-mlp"),
            total_t=add_bones_total_t,
            # pose mlp
            mlp_hidden_dims=getattr(
                self, "ADD_BONES_MLP_HIDDEN_DIMS", [256, 256, 256, 256]
            ),
            pose_dim=23 * 3 if self.mode == "human" else 34 * 3 + 7,
        )

        model = GaussianTemplateModel(
            template=template,
            add_bones=add_bones,
            w_correction_flag=getattr(self, "W_CORRECTION_FLAG", False),
            # w_rest_dim=getattr(self, "W_REST_DIM", 0),
            f_localcode_dim=getattr(self, "F_LOCALCODE_DIM", 0),
            max_sph_order=getattr(self, "MAX_SPH_ORDER", 0),
            w_memory_type=getattr(self, "W_MEMORY_TYPE", "point"),
            max_scale=getattr(self, "MAX_SCALE", 0.1),
            min_scale=getattr(self, "MIN_SCALE", 0.0),
            # * init
            init_mode=getattr(self, "INIT_MODE", "on_mesh"),
            opacity_init_value=getattr(self, "OPACITY_INIT_VALUE", 0.9),
            # on mesh init
            onmesh_init_subdivide_num=getattr(self, "ONMESH_INIT_SUBDIVIDE_NUM", 0),
            onmesh_init_scale_factor=getattr(self, "ONMESH_INIT_SCALE_FACTOR", 1.0),
            onmesh_init_thickness_factor=getattr(
                self, "ONMESH_INIT_THICKNESS_FACTOR", 0.5
            ),
            # near mesh init
            nearmesh_init_num=getattr(self, "NEARMESH_INIT_NUM", 1000),
            nearmesh_init_std=getattr(self, "NEARMESH_INIT_STD", 0.5),
            scale_init_value=getattr(self, "SCALE_INIT_VALUE", 1.0),
        ).to(self.device)

        logging.info(f"Init with {model.N} Gaussians")

        # * set optimizer
        LR_SPH_REST = getattr(self, "LR_SPH_REST", self.LR_SPH / 20.0)
        optimizer = torch.optim.Adam(
            model.get_optimizable_list(
                lr_p=self.LR_P,
                lr_o=self.LR_O,
                lr_s=self.LR_S,
                lr_q=self.LR_Q,
                lr_sph=self.LR_SPH,
                lr_sph_rest=LR_SPH_REST,
                lr_w=self.LR_W,
                lr_w_rest=self.LR_W_REST,
                lr_f=getattr(self, "LR_F_LOCAL", 0.0),
            ),
        )

        xyz_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_P,
            lr_final=self.LR_P_FINAL,
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )
        w_dc_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W,
            lr_final=getattr(self, "LR_W_FINAL", self.LR_W),
            lr_delay_mult=0.01,  # 0.02
        )
        w_rest_scheduler_func = get_expon_lr_func_interval(
            init_step=getattr(self, "W_START_STEP", 0),
            final_step=getattr(self, "W_END_STEP", self.TOTAL_steps),
            lr_init=self.LR_W_REST,
            lr_final=getattr(self, "LR_W_REST_FINAL", self.LR_W_REST),
            lr_delay_mult=0.01,  # 0.02
        )
        sph_scheduler_func = get_expon_lr_func(
            lr_init=self.LR_SPH,
            lr_final=getattr(self, "LR_SPH_FINAL", self.LR_SPH),
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )
        sph_rest_scheduler_func = get_expon_lr_func(
            lr_init=LR_SPH_REST,
            lr_final=getattr(self, "LR_SPH_REST_FINAL", LR_SPH_REST),
            lr_delay_mult=0.01,  # 0.02
            max_steps=self.TOTAL_steps,
        )

        return (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        )

    def _get_pose_optimizer(self, data_provider, add_bones):
        if data_provider is None and add_bones.num_bones == 0:
            # dummy one
            return torch.optim.Adam(params=[torch.zeros(1, requires_grad=True)]), {}

        # * prepare pose optimizer list and the schedulers
        scheduler_dict, pose_optim_l = {}, []
        if data_provider is not None:
            start_step = getattr(self, "POSE_OPTIMIZE_START_STEP", 0)
            end_step = getattr(self, "POSE_OPTIMIZE_END_STEP", self.TOTAL_steps)
            pose_optim_l.extend(
                [
                    {
                        "params": data_provider.pose_base_list,
                        "lr": self.POSE_R_BASE_LR,
                        "name": "pose_base",
                    },
                    {
                        "params": data_provider.pose_rest_list,
                        "lr": self.POSE_R_REST_LR,
                        "name": "pose_rest",
                    },
                    {
                        "params": data_provider.global_trans_list,
                        "lr": self.POSE_T_LR,
                        "name": "pose_trans",
                    },
                ]
            )
            scheduler_dict["pose_base"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_BASE_LR,
                lr_final=getattr(self, "POSE_R_BASE_LR_FINAL", self.POSE_R_BASE_LR),
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_rest"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_R_REST_LR,
                lr_final=getattr(self, "POSE_R_REST_LR_FINAL", self.POSE_R_REST_LR),
                lr_delay_mult=0.01,  # 0.02
            )
            scheduler_dict["pose_trans"] = get_expon_lr_func_interval(
                init_step=start_step,
                final_step=end_step,
                lr_init=self.POSE_T_LR,
                lr_final=getattr(self, "POSE_T_LR_FINAL", self.POSE_T_LR),
                lr_delay_mult=0.01,  # 0.02
            )
        if add_bones.num_bones > 0:
            # have additional bones
            pose_optim_l.append(
                {
                    "params": add_bones.parameters(),
                    "lr": getattr(self, "LR_W_REST_BONES", 0.0),
                    "name": "add_bones",
                }
            )

        pose_optim_mode = getattr(self, "POSE_OPTIM_MODE", "adam")
        if pose_optim_mode == "adam":
            optimizer_smpl = torch.optim.Adam(pose_optim_l)
        elif pose_optim_mode == "sgd":
            optimizer_smpl = torch.optim.SGD(pose_optim_l)
        else:
            raise NotImplementedError(f"Unknown pose optimizer mode {pose_optim_mode}")
        return optimizer_smpl, scheduler_dict

    def is_pose_step(self, step):
        flag = (
            step >= self.POSE_START_STEP
            and step <= self.POSE_END_STEP
            and step % self.POSE_STEP_INTERVAL == 0
        )
        return flag

    def _guide_step(
        self,
        model,
        data_pack,
        act_sph_ord,
        guidance,
        step_ratio,
        guidance_scale,
        head_prob=0.3,
        hand_prob=0.0,
        can_prob=0.0,
    ):
        mvdream_flag = isinstance(guidance, MVDream)
        if mvdream_flag:
            nview = 4
        else:
            nview = 1
        if getattr(self, "RAND_BG_FLAG", False):
            bg = np.random.uniform(0.0, 1.0, size=3)
        else:
            bg = np.array(getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]))
        H, W = 256, 256

        random_seed = np.random.uniform(0.0, 1.0)
        canonical_pose_flag = random_seed < can_prob

        random_seed = np.random.uniform(0.0, 1.0)
        head_flag = random_seed < head_prob and mvdream_flag
        hand_flag = (
            random_seed < head_prob + hand_prob and mvdream_flag and not head_flag
        )

        if hand_flag:
            canonical_pose_flag = True

        pose, trans_zero = data_pack
        trans_zero = torch.zeros_like(trans_zero).to(self.device)
        if canonical_pose_flag:
            _pose = model.template.canonical_pose.detach()
            pose = matrix_to_axis_angle(_pose)[None]
        else:
            pose = pose.to(self.device)
        pose_root = self.reg_base_R_global[None, None]
        pose = torch.cat([pose_root, pose[:, 1:]], dim=1)

        # sample camera
        T_ocam, fovy_deg = sample_camera(
            random_elevation_range=getattr(self, "CAM_ELE_RANGE", [-30, 60]),
            camera_distance_range=getattr(self, "CAM_DIST_RANGE", [1.1, 1.3]),
            relative_radius=True,
            n_view=nview,
        )
        T_ocam = T_ocam.to(self.device).float()
        T_camo = torch.inverse(T_ocam)
        K = fov2K(fovy_deg.to(self.device).float(), H, W)

        def _inner_step(guidance, T_camo, K, pose):
            # this is a pose in canonical ThreeStudio frame, x is forward, y is leftward, z is upward
            mu, frame, s, o, sph, additional_ret = model(
                pose,
                trans_zero,
                additional_dict={},
                active_sph_order=act_sph_ord,
            )
            # convert the mu and frame to camera frame
            mu, frame = transform_mu_frame(mu, frame, T_camo)
            render_pkg_list = []
            for vid in range(nview):
                # this is four view inherited from MVDream
                render_pkg = render_cam_pcl(
                    mu[vid],
                    frame[vid],
                    s[0],
                    o[0],
                    sph[0],
                    H,
                    W,
                    K[vid],
                    False,
                    act_sph_ord,
                    bg,
                )
                render_pkg_list.append(render_pkg)

            rendered_list = [pkg["rgb"] for pkg in render_pkg_list]
            rendered_list = torch.stack(rendered_list, dim=0)  # NVIEW,3,H,W
            if mvdream_flag:
                if guidance.dtype == torch.float16:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        loss = guidance.train_step(
                            rendered_list,
                            opencv2blender(T_ocam),
                            step_ratio=step_ratio,
                            guidance_scale=guidance_scale,
                        )
                else:
                    loss = guidance.train_step(
                        rendered_list,
                        opencv2blender(T_ocam),
                        step_ratio=step_ratio,
                        guidance_scale=guidance_scale,
                    )
            else:
                loss = guidance.train_step(
                    rendered_list,
                    step_ratio=step_ratio,
                    guidance_scale=guidance_scale,
                )
            return loss, render_pkg_list

        if head_flag:
            old_pos_embeds = guidance.pos_embeddings.clone()
            prp = guidance.prompts[0].split(",")[0]
            local_prompt = "Part of a photo, the Head of " + prp
            new_pos_embeds = guidance._encode_text([local_prompt])
            guidance.pos_embeddings = new_pos_embeds

            pose_rot = axis_angle_to_matrix(pose)
            can_template_output = model.template._template_layer(
                body_pose=pose_rot[:, 1:], global_orient=pose_rot[:, 0]
            )
            joint_xyz = can_template_output.joints[0]
            head_xyz = joint_xyz[15]
            T_ocam, fovy_deg = sample_camera(
                random_elevation_range=[-10.0, 20.0],
                camera_distance_range=[0.25, 0.25],
                relative_radius=True,
                # random_azimuth_range=[0.0,0.0],
                fovy_range=[15, 60],
                zoom_range=[1.0, 1.0],
            )
            T_ocam[:, :3, -1] += head_xyz[None].to(T_ocam.device)
            T_ocam = T_ocam.to(self.device).float()
            T_camo = torch.inverse(T_ocam)
            K = fov2K(fovy_deg.to(self.device).float(), H, W)
            loss, render_pkg_list = _inner_step(guidance, T_camo, K, pose)

            # viz_rgb = [it["rgb"].permute(1,2,0) for it in render_pkg_list_head]
            # viz_rgb = torch.cat(viz_rgb, dim=1)
            # imageio.imsave("./debug/viz_head.png", viz_rgb.detach().cpu().numpy())
            guidance.pos_embeddings = old_pos_embeds
        elif hand_flag:
            random_seed = np.random.uniform(0.0, 1.0)
            left_hand_flag = random_seed < 0.5
            if left_hand_flag:
                joint_id = 22
                hand_str = "left"
            else:
                joint_id = 23
                hand_str = "right"

            old_pos_embeds = guidance.pos_embeddings.clone()
            prp = guidance.prompts[0].split(",")[0]
            local_prompt = f"Part of a photo, zoomed in, the {hand_str} hand of " + prp
            new_pos_embeds = guidance._encode_text([local_prompt])
            guidance.pos_embeddings = new_pos_embeds

            pose_rot = axis_angle_to_matrix(pose)
            can_template_output = model.template._template_layer(
                body_pose=pose_rot[:, 1:], global_orient=pose_rot[:, 0]
            )
            joint_xyz = can_template_output.joints[0]
            head_xyz = joint_xyz[joint_id]

            T_ocam, fovy_deg = sample_camera(
                random_elevation_range=[-30.0, 30.0],
                camera_distance_range=[0.25, 0.25],
                relative_radius=True,
                fovy_range=[15, 60],
                zoom_range=[1.0, 1.0],
                random_azimuth_range=[-180.0 - 30, 0 + 30]
                if joint_id == 23
                else [0 - 30, 180 + 30],
            )
            T_ocam[:, :3, -1] += head_xyz[None].to(T_ocam.device)
            T_ocam = T_ocam.to(self.device).float()
            T_camo = torch.inverse(T_ocam)
            K = fov2K(fovy_deg.to(self.device).float(), H, W)
            loss, render_pkg_list = _inner_step(guidance, T_camo, K, pose)

            # viz_rgb = [it["rgb"].permute(1, 2, 0) for it in render_pkg_list]
            # viz_rgb = torch.cat(viz_rgb, dim=1)
            # imageio.imsave("./debug/viz_hand.png", viz_rgb.detach().cpu().numpy())
            guidance.pos_embeddings = old_pos_embeds
        else:
            loss, render_pkg_list = _inner_step(guidance, T_camo, K, pose)
        return loss, render_pkg_list

    def _fit_step(
        self,
        model,
        data_pack,
        act_sph_ord,
        random_bg=True,
        scale_multiplier=1.0,
        opacity_multiplier=1.0,
        opa_th=-1,
        use_box_crop_pad=-1,
        default_bg=[1.0, 1.0, 1.0],
        add_bones_As=None,
    ):
        gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index = data_pack
        gt_rgb = gt_rgb.clone()

        pose = torch.cat([pose_base, pose_rest], dim=1)
        H, W = gt_rgb.shape[1:3]
        additional_dict = {"t": time_index}
        if add_bones_As is not None:
            additional_dict["As"] = add_bones_As
        mu, fr, sc, op, sph, additional_ret = model(
            pose,
            global_trans,
            additional_dict=additional_dict,
            active_sph_order=act_sph_ord,
        )

        sc = sc * scale_multiplier
        op = op * opacity_multiplier

        loss_recon = 0.0
        loss_lpips, loss_ssim = (
            torch.zeros(1).to(gt_rgb.device).squeeze(),
            torch.zeros(1).to(gt_rgb.device).squeeze(),
        )
        loss_mask = 0.0
        render_pkg_list, rgb_target_list, mask_target_list = [], [], []
        for i in range(len(gt_rgb)):
            if random_bg:
                bg = np.random.uniform(0.0, 1.0, size=3)
            else:
                bg = np.array(default_bg)
            bg_tensor = torch.from_numpy(bg).float().to(gt_rgb.device)
            gt_rgb[i][gt_mask[i] == 0] = bg_tensor[None, None, :]
            render_pkg = render_cam_pcl(
                mu[i], fr[i], sc[i], op[i], sph[i], H, W, K[i], False, act_sph_ord, bg
            )
            if opa_th > 0.0:
                bg_mask = render_pkg["alpha"][0] < opa_th
                render_pkg["rgb"][:, bg_mask] = (
                    render_pkg["rgb"][:, bg_mask] * 0.0 + default_bg[0]
                )

            # * lpips before the pad
            if self.LAMBDA_LPIPS > 0:
                _loss_lpips = self.lpips(
                    render_pkg["rgb"][None],
                    gt_rgb.permute(0, 3, 1, 2),
                )
                loss_lpips = loss_lpips + _loss_lpips

            if use_box_crop_pad > 0:
                # pad the gt mask and crop the image, for a large image with a small fg object
                _yl, _yr, _xl, _xr = get_bbox(gt_mask[i], use_box_crop_pad)
                for key in ["rgb", "alpha", "dep"]:
                    render_pkg[key] = render_pkg[key][:, _yl:_yr, _xl:_xr]
                rgb_target = gt_rgb[i][_yl:_yr, _xl:_xr]
                mask_target = gt_mask[i][_yl:_yr, _xl:_xr]
            else:
                rgb_target = gt_rgb[i]
                mask_target = gt_mask[i]

            # * standard recon loss
            _loss_recon = abs(render_pkg["rgb"].permute(1, 2, 0) - rgb_target).mean()
            _loss_mask = abs(render_pkg["alpha"].squeeze(0) - mask_target).mean()

            # * ssim
            if self.LAMBDA_SSIM > 0:
                _loss_ssim = 1.0 - ssim(
                    render_pkg["rgb"][None],
                    rgb_target.permute(2, 0, 1)[None],
                )
                loss_ssim = loss_ssim + _loss_ssim

            loss_recon = loss_recon + _loss_recon
            loss_mask = loss_mask + _loss_mask
            render_pkg_list.append(render_pkg)
            rgb_target_list.append(rgb_target)
            mask_target_list.append(mask_target)

        loss_recon = loss_recon / len(gt_rgb)
        loss_mask = loss_mask / len(gt_rgb)
        loss_lpips = loss_lpips / len(gt_rgb)
        loss_ssim = loss_ssim / len(gt_rgb)

        loss = (
            loss_recon + self.LAMBDA_SSIM * loss_ssim + self.LAMBDA_LPIPS * loss_lpips
        )

        return (
            loss,
            loss_mask,
            render_pkg_list,
            rgb_target_list,
            mask_target_list,
            (mu, fr, sc, op, sph),
            {
                "loss_l1_recon": loss_recon,
                "loss_lpips": loss_lpips,
                "loss_ssim": loss_ssim,
            },
        )

    def compute_reg3D(self, model: GaussianTemplateModel):
        K = getattr(self, "CANONICAL_SPACE_REG_K", 10)
        (
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
            dist_sq,
            max_s_sq,
        ) = model.compute_reg(K)

        lambda_std_q = getattr(self, "LAMBDA_STD_Q", 0.0)
        lambda_std_s = getattr(self, "LAMBDA_STD_S", 0.0)
        lambda_std_o = getattr(self, "LAMBDA_STD_O", 0.0)
        lambda_std_cd = getattr(self, "LAMBDA_STD_CD", 0.0)
        lambda_std_ch = getattr(self, "LAMBDA_STD_CH", lambda_std_cd)
        lambda_std_w = getattr(self, "LAMBDA_STD_W", 0.3)
        lambda_std_w_rest = getattr(self, "LAMBDA_STD_W_REST", lambda_std_w)
        lambda_std_f = getattr(self, "LAMBDA_STD_F", lambda_std_w)

        lambda_w_norm = getattr(self, "LAMBDA_W_NORM", 0.1)
        lambda_w_rest_norm = getattr(self, "LAMBDA_W_REST_NORM", lambda_w_norm)

        lambda_knn_dist = getattr(self, "LAMBDA_KNN_DIST", 0.0)

        lambda_small_scale = getattr(self, "LAMBDA_SMALL_SCALE", 0.0)

        reg_loss = (
            lambda_std_q * q_std
            + lambda_std_s * s_std
            + lambda_std_o * o_std
            + lambda_std_cd * cd_std
            + lambda_std_ch * ch_std
            + lambda_knn_dist * dist_sq
            + lambda_std_w * w_std
            + lambda_std_w_rest * w_rest_std
            + lambda_std_f * f_std
            + lambda_w_norm * w_norm
            + lambda_w_rest_norm * w_rest_norm
            + lambda_small_scale * max_s_sq
        )

        details = {
            "q_std": q_std.detach(),
            "s_std": s_std.detach(),
            "o_std": o_std.detach(),
            "cd_std": cd_std.detach(),
            "ch_std": ch_std.detach(),
            "w_std": w_std.detach(),
            "w_rest_std": w_rest_std.detach(),
            "f_std": f_std.detach(),
            "knn_dist_sq": dist_sq.detach(),
            "w_norm": w_norm.detach(),
            "w_rest_norm": w_rest_norm.detach(),
            "max_s_sq": max_s_sq.detach(),
        }
        return reg_loss, details

    def add_scalar(self, *args, **kwargs):
        if self.FAST_TRAINING:
            return
        if getattr(self, "NO_TB", False):
            return
        self.writer.add_scalar(*args, **kwargs)
        return

    def get_sd_step_ratio(self, step):
        start = getattr(self, "SD_RATIO_START_STEP", 0)
        end = getattr(self, "SD_RATIO_END_STEP", self.TOTAL_steps)
        len = end - start
        if (step + 1) <= start:
            return 1.0 / len
        if (step + 1) >= end:
            return 1.0
        ratio = min(1, (step - start + 1) / len)
        ratio = max(1.0 / len, ratio)
        return ratio

    def get_guidance_scale(self, step):
        scale = self.GUIDANCE_SCALE
        end_scale = getattr(self, "GUIDANCE_SCALE_END", scale)
        ret = scale + (end_scale - scale) * (step / self.TOTAL_steps)
        return ret

    def run(
        self,
        real_data_provider=None,
        fake_data_provider=None,
        test_every_n_step=-1,
        test_dataset=None,
        guidance=None,
    ):
        torch.cuda.empty_cache()

        # * init the model
        real_flag = real_data_provider is not None
        fake_flag = fake_data_provider is not None
        assert real_flag or fake_flag

        if real_flag:
            init_beta = real_data_provider.betas
            total_t = real_data_provider.total_t
        else:
            init_beta = np.zeros(10)
            total_t = None
            assert guidance is not None
            # self.amp_scaler = torch.cuda.amp.GradScaler()
        (
            model,
            optimizer,
            xyz_scheduler_func,
            w_dc_scheduler_func,
            w_rest_scheduler_func,
            sph_scheduler_func,
            sph_rest_scheduler_func,
        ) = self._get_model_optimizer(betas=init_beta, add_bones_total_t=total_t)

        optimizer_pose, scheduler_pose = self._get_pose_optimizer(
            real_data_provider, model.add_bones
        )

        # * Optimization Loop
        stat_n_list = []
        active_sph_order, last_reset_step = 0, -1
        seed_everything(self.SEED)
        running_start_t = time.time()
        logging.info(f"Start training at {running_start_t}")
        for step in tqdm(range(self.TOTAL_steps)):
            update_learning_rate(xyz_scheduler_func(step), "xyz", optimizer)
            update_learning_rate(sph_scheduler_func(step), "f_dc", optimizer)
            update_learning_rate(sph_rest_scheduler_func(step), "f_rest", optimizer)

            update_learning_rate(
                w_dc_scheduler_func(step), ["w_dc", "w_dc_vox"], optimizer
            )
            update_learning_rate(
                w_rest_scheduler_func(step), ["w_rest", "w_rest_vox"], optimizer
            )
            for k, v in scheduler_pose.items():
                update_learning_rate(v(step), k, optimizer_pose)

            if step in self.INCREASE_SPH_STEP:
                active_sph_order += 1
                logging.info(f"active_sph_order: {active_sph_order}")

            # * Recon fitting step
            model.train()
            optimizer.zero_grad(), optimizer_pose.zero_grad()

            loss = 0.0
            if real_flag:
                real_data_pack = real_data_provider(
                    self.N_POSES_PER_STEP, continuous=False
                )
                (
                    loss_recon,
                    loss_mask,
                    render_list,
                    gt_rgb,
                    gt_mask,
                    model_ret,
                    loss_dict,
                ) = self._fit_step(
                    model,
                    data_pack=real_data_pack,
                    act_sph_ord=active_sph_order,
                    random_bg=self.RAND_BG_FLAG,
                    use_box_crop_pad=getattr(self, "BOX_CROP_PAD", -1),
                    default_bg=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                )
                loss = loss + loss_recon
                for k, v in loss_dict.items():
                    self.add_scalar(k, v.detach(), step)
            else:
                render_list = []
            if fake_flag:
                # do SD guidance enhancement
                fake_data_pack = fake_data_provider(self.N_POSES_PER_GUIDE_STEP)
                # * get step ratio
                sd_step_ratio = self.get_sd_step_ratio(step)
                guidance_scale = self.get_guidance_scale(step)

                loss_guidance, guide_rendered_list = self._guide_step(
                    model,
                    fake_data_pack,
                    active_sph_order,
                    guidance,
                    step_ratio=sd_step_ratio,
                    guidance_scale=guidance_scale,
                    head_prob=getattr(self, "HEAD_PROB", 0.3),
                    can_prob=getattr(self, "CAN_PROB", 0.0),
                    hand_prob=getattr(self, "HAND_PROB", 0.0),
                )
                lambda_guidance = getattr(self, "LAMBDA_GUIDANCE", 1.0)
                loss = loss + lambda_guidance * loss_guidance
                self.add_scalar("loss_guide", loss_guidance.detach(), step)
                render_list = render_list + guide_rendered_list

            if (
                last_reset_step < 0
                or step - last_reset_step > self.MASK_LOSS_PAUSE_AFTER_RESET
                and step > getattr(self, "MASK_START_STEP", 0)
            ) and real_flag:
                loss = loss + self.LAMBDA_MASK * loss_mask
                self.add_scalar("loss_mask", loss_mask.detach(), step)
            self.add_scalar("loss", loss.detach(), step)

            # * Reg Terms
            reg_loss, reg_details = self.compute_reg3D(model)
            loss = reg_loss + loss
            for k, v in reg_details.items():
                self.add_scalar(k, v.detach(), step)

            loss.backward()
            optimizer.step()

            if step > getattr(self, "POSE_OPTIMIZE_START_STEP", -1):
                optimizer_pose.step()

            self.add_scalar("N", model.N, step)

            # * Gaussian Control
            if step > self.DENSIFY_START:
                for render_pkg in render_list:
                    model.record_xyz_grad_radii(
                        render_pkg["viewspace_points"],
                        render_pkg["radii"],
                        render_pkg["visibility_filter"],
                    )

            if (
                step > self.DENSIFY_START
                and step < getattr(self, "DENSIFY_END", 10000000)
                and step % self.DENSIFY_INTERVAL == 0
            ):
                N_old = model.N
                model.densify(
                    optimizer=optimizer,
                    max_grad=self.MAX_GRAD,
                    percent_dense=self.PERCENT_DENSE,
                    extent=0.5,
                    verbose=True,
                )
                logging.info(f"Densify: {N_old}->{model.N}")

            if step > self.PRUNE_START and step % self.PRUNE_INTERVAL == 0:
                N_old = model.N
                model.prune_points(
                    optimizer,
                    min_opacity=self.OPACIT_PRUNE_TH,
                    max_screen_size=1e10,  # ! disabled
                )
                logging.info(f"Prune: {N_old}->{model.N}")
            if step in getattr(self, "RANDOM_GROW_STEPS", []):
                model.random_grow(
                    optimizer,
                    num_factor=getattr(self, "NUM_FACTOR", 0.1),
                    std=getattr(self, "RANDOM_GROW_STD", 0.1),
                    init_opa_value=getattr(self, "RANDOM_GROW_OPA", 0.01),
                )

            if (step + 1) in self.RESET_OPACITY_STEPS:
                model.reset_opacity(optimizer, self.OPACIT_RESET_VALUE)
                last_reset_step = step

            if (step + 1) in getattr(self, "REGAUSSIAN_STEPS", []):
                model.regaussian(optimizer, self.REGAUSSIAN_STD)

            stat_n_list.append(model.N)
            if self.FAST_TRAINING:
                continue

            # * Viz
            if (step + 1) % getattr(self, "VIZ_INTERVAL", 100) == 0 or step == 0:
                if real_flag:
                    mu, fr, s, o, sph = model_ret[:5]
                    save_path = f"{self.log_dir}/viz_step/step_{step}.png"
                    viz_render(
                        gt_rgb[0], gt_mask[0], render_list[0], save_path=save_path
                    )
                    # viz the spinning in the middle
                    (
                        _,
                        _,
                        K,
                        pose_base,
                        pose_rest,
                        global_trans,
                        time_index,
                    ) = real_data_pack
                    viz_spinning(
                        model,
                        torch.cat([pose_base, pose_rest], 1)[:1],
                        global_trans[:1],
                        real_data_provider.H,
                        real_data_provider.W,
                        K[0],
                        save_path=f"{self.log_dir}/viz_step/spinning_{step}.gif",
                        time_index=time_index,
                        active_sph_order=active_sph_order,
                        # bg_color=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                        bg_color=[1.0, 1.0, 1.0],
                    )
                if fake_flag:
                    viz_rgb = [it["rgb"].permute(1, 2, 0) for it in guide_rendered_list]
                    viz_rgb = torch.cat(viz_rgb, dim=1).detach().cpu().numpy()
                    viz_rgb = (np.clip(viz_rgb, 0.0, 1.0) * 255).astype(np.uint8)
                    imageio.imsave(f"{self.log_dir}/viz_step/guide_{step}.png", viz_rgb)

                can_pose = model.template.canonical_pose.detach()
                if self.mode == "human":
                    can_pose[0] = self.viz_base_R.to(can_pose.device)
                    can_pose = matrix_to_axis_angle(can_pose)[None]
                else:
                    can_pose[:3] = matrix_to_axis_angle(self.viz_base_R[None])[0]
                    can_pose = can_pose[None]
                can_trans = torch.zeros(len(can_pose), 3).to(can_pose)
                can_trans[:, -1] = 3.0
                viz_H, viz_W = 512, 512
                viz_K = fov2K(60, viz_H, viz_W)
                viz_spinning(
                    model,
                    can_pose,
                    can_trans,
                    viz_H,
                    viz_W,
                    viz_K,
                    save_path=f"{self.log_dir}/viz_step/spinning_can_{step}.gif",
                    time_index=None,  # canonical pose use t=None
                    active_sph_order=active_sph_order,
                    # bg_color=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                    bg_color=[1.0, 1.0, 1.0],
                )

                # viz the distrbution
                plt.figure(figsize=(15, 3))
                scale = model.get_s
                for i in range(3):
                    _s = scale[:, i].detach().cpu().numpy()
                    plt.subplot(1, 3, i + 1)
                    plt.hist(_s, bins=100)
                    plt.title(f"scale {i}")
                    plt.grid(), plt.ylabel("count"), plt.xlabel("scale")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/s_hist_step_{step}.png")
                plt.close()

                plt.figure(figsize=(12, 4))
                opacity = (model.get_o).squeeze(-1)
                plt.hist(opacity.detach().cpu().numpy(), bins=100)
                plt.title(f"opacity")
                plt.grid(), plt.ylabel("count"), plt.xlabel("opacity")
                plt.tight_layout()
                plt.savefig(f"{self.log_dir}/viz_step/o_hist_step_{step}.png")
                plt.close()

            # * test
            if test_every_n_step > 0 and (
                (step + 1) % test_every_n_step == 0 or step == self.TOTAL_steps - 1
            ):
                logging.info("Testing...")
                self._eval_save_error_map(model, test_dataset, f"test_{step}")
                test_results = self.eval_dir(f"test_{step}")
                for k, v in test_results.items():
                    self.add_scalar(f"test_{k}", v, step)

        running_end_t = time.time()
        logging.info(
            f"Training time: {(running_end_t - running_start_t):.3f} seconds i.e. {(running_end_t - running_start_t)/60.0 :.3f} minutes"
        )

        # * save
        logging.info("Saving model...")
        model.eval()
        ckpt_path = f"{self.log_dir}/model.pth"
        torch.save(model.state_dict(), ckpt_path)
        if real_flag:
            pose_path = f"{self.log_dir}/training_poses.pth"
            torch.save(real_data_provider.state_dict(), pose_path)
        model.to("cpu")

        # * stat
        plt.figure(figsize=(5, 5))
        plt.plot(stat_n_list)
        plt.title("N"), plt.xlabel("step"), plt.ylabel("N")
        plt.savefig(f"{self.log_dir}/N.png")
        plt.close()

        model.summary()
        return model, real_data_provider

    def testtime_pose_optimization(
        self,
        data_pack,
        model,
        evaluator,
        pose_base_lr=3e-3,
        pose_rest_lr=3e-3,
        trans_lr=3e-3,
        steps=100,
        decay_steps=30,
        decay_factor=0.5,
        check_every_n_step=5,
        viz_fn=None,
        As=None,
        pose_As_lr=1e-3,
    ):
        # * Like Instant avatar, optimize the smpl pose f
        # * will optimize all poses in the data_pack
        torch.cuda.empty_cache()
        seed_everything(self.SEED)
        model.eval()  # to get gradients, but never optimized
        evaluator.eval()
        gt_rgb, gt_mask, K, pose_b, pose_r, trans = data_pack[:6]
        pose_b = pose_b.detach().clone()
        pose_r = pose_r.detach().clone()
        trans = trans.detach().clone()
        pose_b.requires_grad_(True)
        pose_r.requires_grad_(True)
        trans.requires_grad_(True)
        gt_rgb, gt_mask = gt_rgb.to(self.device), gt_mask.to(self.device)

        optim_l = [
            {"params": [pose_b], "lr": pose_base_lr},
            {"params": [pose_r], "lr": pose_rest_lr},
            {"params": [trans], "lr": trans_lr},
        ]
        if As is not None:
            As = As.detach().clone()
            As.requires_grad_(True)
            optim_l.append({"params": [As], "lr": pose_As_lr})
        optimizer_smpl = torch.optim.SGD(optim_l)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_smpl, step_size=decay_steps, gamma=decay_factor
        )

        loss_list, psnr_list, ssim_list, lpips_list = [], [], [], []
        viz_step_list = []
        for inner_step in range(steps):
            # optimize
            optimizer_smpl.zero_grad()
            loss_recon, _, rendered_list, _, _, _, _ = self._fit_step(
                model,
                [gt_rgb, gt_mask, K, pose_b, pose_r, trans, None],
                act_sph_ord=model.max_sph_order,
                random_bg=False,
                default_bg=getattr(self, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                add_bones_As=As,
            )
            loss = loss_recon
            loss.backward()
            optimizer_smpl.step()
            scheduler.step()
            loss_list.append(float(loss))

            if (
                inner_step % check_every_n_step == 0
                or inner_step == steps - 1
                and viz_fn is not None
            ):
                viz_step_list.append(inner_step)
                with torch.no_grad():
                    _check_results = [
                        evaluator(
                            rendered_list[0]["rgb"].permute(1, 2, 0)[None],
                            gt_rgb[0][None],
                        )
                    ]
                psnr = torch.stack([r["psnr"] for r in _check_results]).mean().item()
                ssim = torch.stack([r["ssim"] for r in _check_results]).mean().item()
                lpips = torch.stack([r["lpips"] for r in _check_results]).mean().item()
                psnr_list.append(float(psnr))
                ssim_list.append(float(ssim))
                lpips_list.append(float(lpips))

        if viz_fn is not None:
            os.makedirs(osp.dirname(viz_fn), exist_ok=True)
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 4, 1)
            plt.plot(viz_step_list, psnr_list)
            plt.title(f"PSNR={psnr_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 2)
            plt.plot(viz_step_list, ssim_list)
            plt.title(f"SSIM={ssim_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 3)
            plt.plot(viz_step_list, lpips_list)
            plt.title(f"LPIPS={lpips_list[-1]}"), plt.grid()
            plt.subplot(1, 4, 4)
            plt.plot(loss_list)
            plt.title(f"Loss={loss_list[-1]}"), plt.grid()
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(viz_fn)
            plt.close()

        if As is not None:
            As.detach().clone()
        return (
            pose_b.detach().clone(),
            pose_r.detach().clone(),
            trans.detach().clone(),
            As,
        )

    @torch.no_grad()
    def eval_fps(self, model, real_data_provider, rounds=1):
        model.eval()
        model.cache_for_fast()
        logging.info(f"Model has {model.N} points.")
        N_frames = len(real_data_provider.rgb_list)
        ret = real_data_provider(N_frames)
        gt_rgb, gt_mask, K, pose_base, pose_rest, global_trans, time_index = ret
        pose = torch.cat([pose_base, pose_rest], 1)
        H, W = gt_rgb.shape[1:3]
        sph_o = model.max_sph_order
        logging.info(f"FPS eval using active_sph_order: {sph_o}")
        # run one iteration to check the output correctness

        mu, fr, sc, op, sph, additional_ret = model(
            pose[0:1],
            global_trans[0:1],
            additional_dict={},
            active_sph_order=sph_o,
            fast=True,
        )
        bg = [1.0, 1.0, 1.0]
        render_pkg = render_cam_pcl(
            mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
        )
        pred = render_pkg["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        imageio.imsave(osp.join(self.log_dir, "fps_eval_sample.png"), pred)

        logging.info("Start FPS test...")
        start_t = time.time()

        for j in tqdm(range(int(N_frames * rounds))):
            i = j % N_frames
            mu, fr, sc, op, sph, additional_ret = model(
                pose[i : i + 1],
                global_trans[i : i + 1],
                additional_dict={"t": time_index},
                active_sph_order=sph_o,
                fast=True,
            )
            bg = [1.0, 1.0, 1.0]
            render_pkg = render_cam_pcl(
                mu[0], fr[0], sc[0], op[0], sph[0], H, W, K[0], False, sph_o, bg
            )
        end_t = time.time()

        fps = (rounds * N_frames) / (end_t - start_t)
        logging.info(f"FPS: {fps}")
        with open(osp.join(self.log_dir, "fps.txt"), "w") as f:
            f.write(f"FPS: {fps}")
        return fps


def tg_fitting_eval(solver, dataset_mode, seq_name, optimized_seq):
    if dataset_mode == "people_snapshot":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=300,
            tto_decay=70,
            dataset_mode=dataset_mode,
            pose_base_lr=3e-3,
            pose_rest_lr=3e-3,
            trans_lr=3e-3,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "zju":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=50,
            tto_decay=20,
            dataset_mode=dataset_mode,
            pose_base_lr=4e-3,
            pose_rest_lr=4e-3,
            trans_lr=4e-3,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "instant_avatar_wild":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=50,
            tto_decay=20,
            dataset_mode=dataset_mode,
            pose_base_lr=4e-3,
            pose_rest_lr=4e-3,
            trans_lr=4e-3,
            training_optimized_seq=optimized_seq,
        )
    elif dataset_mode == "dog_demo":
        test(
            solver,
            seq_name=seq_name,
            tto_flag=True,
            tto_step=160,
            tto_decay=50,
            dataset_mode=dataset_mode,
            pose_base_lr=2e-2,
            pose_rest_lr=2e-2,
            trans_lr=2e-2,
        )
    else:
        pass
    # solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)
    return


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--profile", type=str, default="./profiles/people/people_1m.yaml")
    args.add_argument("--dataset", type=str, default="people_snapshot")
    args.add_argument("--seq", type=str, default="male-3-casual")
    args.add_argument("--logbase", type=str, default="debug")
    # remove the viz during optimization
    args.add_argument("--fast", action="store_true")
    args.add_argument("--no_eval", action="store_true")
    # for eval
    args.add_argument("--eval_only", action="store_true")
    args.add_argument("--log_dir", default="", type=str)
    args.add_argument("--skip_eval_if_exsists", action="store_true")
    # for viz
    args.add_argument("--viz_only", action="store_true")
    args = args.parse_args()

    device = torch.device("cuda:0")
    dataset_mode = args.dataset
    seq_name = args.seq
    profile_fn = args.profile
    base_name = args.logbase

    if len(args.log_dir) > 0:
        log_dir = args.log_dir
    else:
        os.makedirs(f"./logs/{base_name}", exist_ok=True)
        profile_name = osp.basename(profile_fn).split(".")[0]
        log_dir = (
            f"./logs/{base_name}/seq={seq_name}_prof={profile_name}_data={dataset_mode}"
        )

    # prepare
    if dataset_mode == "zju":
        smpl_path = "./data/smpl-meta/SMPL_NEUTRAL.pkl"
        mode = "human"
    elif dataset_mode == "people_snapshot":
        mode = "human"
        if seq_name.startswith("female"):
            smpl_path = "./data/smpl_model/SMPL_FEMALE.pkl"
        elif seq_name.startswith("male"):
            smpl_path = "./data/smpl_model/SMPL_MALE.pkl"
        else:
            smpl_path = "./data/smpl_model/SMPL_NEUTRAL.pkl"
    elif dataset_mode == "instant_avatar_wild":
        mode = "human"
        smpl_path = "./data/smpl_model/SMPL_NEUTRAL.pkl"
    elif dataset_mode == "ubcfashion":
        mode = "human"
        smpl_path = "./data/smpl_model/SMPL_NEUTRAL.pkl"
    elif dataset_mode == "dog_demo":
        mode = "dog"
        smpl_path = None
    else:
        raise NotImplementedError()

    solver = TGFitter(
        log_dir=log_dir,
        profile_fn=profile_fn,
        mode=mode,
        template_model_path=smpl_path,
        device=device,
        FAST_TRAINING=args.fast,
    )
    data_provider, trainset = solver.prepare_real_seq(
        seq_name,
        dataset_mode,
        split="train",
        ins_avt_wild_start_end_skip=getattr(solver, "START_END_SKIP", None),
    )

    if args.eval_only:
        assert len(args.log_dir) > 0, "Please specify the log_dir for eval only mode"
        assert osp.exists(log_dir), f"{log_dir} not exists!"
        logging.info(f"Eval only mode, load model from {log_dir}")
        if args.skip_eval_if_exsists:
            test_dir = osp.join(log_dir, "test")
            if osp.exists(test_dir):
                logging.info(f"Eval already exists, skip")
                sys.exit(0)

        data_provider.load_state_dict(
            torch.load(osp.join(log_dir, "training_poses.pth")), strict=True
        )
        solver.eval_fps(solver.load_saved_model(), data_provider, rounds=10)
        tg_fitting_eval(solver, dataset_mode, seq_name, data_provider)
        logging.info("Done")
        sys.exit(0)
    elif args.viz_only:
        assert len(args.log_dir) > 0, "Please specify the log_dir for eval only mode"
        assert osp.exists(log_dir), f"{log_dir} not exists!"
        data_provider.load_state_dict(
            torch.load(osp.join(log_dir, "training_poses.pth")), strict=True
        )
        logging.info(f"Viz only mode, load model from {log_dir}")
        if mode == "human":
            viz_human_all(solver, data_provider, viz_name="viz_only")
        elif mode == "dog":
            viz_dog_all(solver, data_provider, viz_name="viz_only")
        logging.info("Done")
        sys.exit(0)

    logging.info(f"Optimization with {profile_fn}")
    _, optimized_seq = solver.run(data_provider)

    if mode == "human":
        viz_human_all(solver, optimized_seq)
    elif mode == "dog":
        viz_dog_all(solver, optimized_seq)

    solver.eval_fps(solver.load_saved_model(), optimized_seq, rounds=10)
    if args.no_eval:
        logging.info("No eval, done!")
        sys.exit(0)

    tg_fitting_eval(solver, dataset_mode, seq_name, optimized_seq)

    logging.info("done!")
