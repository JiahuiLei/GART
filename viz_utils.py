from matplotlib import pyplot as plt
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
import imageio
import torch
from tqdm import tqdm
import numpy as np
import warnings, os, os.path as osp, shutil, sys
from transforms3d.euler import euler2mat
from lib_data.data_provider import RealDataOptimizablePoseProviderPose

from lib_gart.optim_utils import *
from lib_render.gauspl_renderer import render_cam_pcl
from lib_gart.model_utils import transform_mu_frame

from utils.misc import *
from utils.viz import viz_render


@torch.no_grad()
def viz_spinning(
    model,
    pose,
    trans,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    mu, fr, s, o, sph, additional_ret = model(
        pose, trans, {"t": time_index}, active_sph_order=active_sph_order
    )
    if model_mask is not None:
        assert len(model_mask) == mu.shape[1]
        mu = mu[:, model_mask.bool()]
        fr = fr[:, model_mask.bool()]
        s = s[:, model_mask.bool()]
        o = o[:, model_mask.bool()]
        sph = sph[:, model_mask.bool()]

    viz_frames = []
    for vid in range(n_spinning):
        spin_R = (
            torch.from_numpy(euler2mat(0, 2 * np.pi * vid / n_spinning, 0, "sxyz"))
            .to(device)
            .float()
        )
        spin_t = mu.mean(1)[0]
        spin_t = (torch.eye(3).to(device) - spin_R) @ spin_t[:, None]
        spin_T = torch.eye(4).to(device)
        spin_T[:3, :3] = spin_R
        spin_T[:3, 3] = spin_t.squeeze(-1)
        viz_mu, viz_fr = transform_mu_frame(mu, fr, spin_T[None])

        render_pkg = render_cam_pcl(
            viz_mu[0],
            viz_fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


@torch.no_grad()
def viz_spinning_self_rotate(
    model,
    base_R,
    pose,
    trans,
    H,
    W,
    K,
    save_path,
    time_index=None,
    n_spinning=10,
    model_mask=None,
    active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
):
    device = pose.device
    viz_frames = []
    # base_R = base_R.detach().cpu().numpy()
    first_R = axis_angle_to_matrix(pose[:, 0])[0].detach().cpu().numpy()
    for vid in range(n_spinning):
        rotation = euler2mat(0.0, 2 * np.pi * vid / n_spinning, 0.0, "sxyz")
        rotation = torch.from_numpy(first_R @ rotation).float().to(device)
        pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]

        mu, fr, s, o, sph, additional_ret = model(
            pose, trans, {"t": time_index}, active_sph_order=active_sph_order
        )
        if model_mask is not None:
            assert len(model_mask) == mu.shape[1]
            mu = mu[:, model_mask.bool()]
            fr = fr[:, model_mask.bool()]
            s = s[:, model_mask.bool()]
            o = o[:, model_mask.bool()]
            sph = sph[:, model_mask.bool()]

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            active_sph_order,
            bg_color=bg_color,
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


@torch.no_grad()
def viz_human_all(
    solver,
    data_provider: RealDataOptimizablePoseProviderPose = None,
    ckpt_dir=None,
    training_skip=1,
    n_spinning=40,
    novel_pose_dir="novel_poses",
    novel_skip=2,
    model=None,
    model_mask=None,
    viz_name="",
    export_mesh_flag=False,  # remove this from release version
):
    if model is None:
        model = solver.load_saved_model(ckpt_dir)
    model.eval()

    viz_dir = osp.join(solver.log_dir, f"{viz_name}_human_viz")
    os.makedirs(viz_dir, exist_ok=True)

    active_sph_order = int(model.max_sph_order)

    if data_provider is not None:
        # if ckpt_dir is None:
        #     ckpt_dir = solver.log_dir
        # pose_path = osp.join(ckpt_dir, "pose.pth")
        pose_base_list = data_provider.pose_base_list
        pose_rest_list = data_provider.pose_rest_list
        global_trans_list = data_provider.global_trans_list
        pose_list = torch.cat([pose_base_list, pose_rest_list], 1)
        pose_list, global_trans_list = pose_list.to(
            solver.device
        ), global_trans_list.to(solver.device)
        rgb_list = data_provider.rgb_list
        mask_list = data_provider.mask_list
        K_list = data_provider.K_list
        H, W = rgb_list.shape[1:3]
    else:
        H, W = 512, 512
        K_list = [torch.from_numpy(fov2K(45, H, W)).float().to(solver.device)]
        global_trans_list = torch.zeros(1, 3).to(solver.device)
        global_trans_list[0, -1] = 3.0

    # viz training
    if data_provider is not None:
        print("Viz training...")
        viz_frames = []
        for t in range(len(pose_list)):
            if t % training_skip != 0:
                continue
            pose = pose_list[t][None]
            K = K_list[t]
            trans = global_trans_list[t][None]
            time_index = torch.Tensor([t]).long().to(solver.device)
            mu, fr, s, o, sph, _ = model(
                pose,
                trans,
                {"t": time_index},  # use time_index from training set
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                sph[0],
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
            )
            viz_frame = viz_render(rgb_list[t], mask_list[t], render_pkg)
            viz_frames.append(viz_frame)
        imageio.mimsave(f"{viz_dir}/training.gif", viz_frames)

    # viz static spinning
    print("Viz spinning...")
    can_pose = model.template.canonical_pose.detach()
    viz_base_R_opencv = np.asarray(euler2mat(np.pi, 0, 0, "sxyz"))
    viz_base_R_opencv = torch.from_numpy(viz_base_R_opencv).float()
    can_pose[0] = viz_base_R_opencv.to(can_pose.device)
    can_pose = matrix_to_axis_angle(can_pose)[None]
    dapose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    dapose[:, 1, -1] = np.pi / 4
    dapose[:, 2, -1] = -np.pi / 4
    dapose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    tpose = torch.from_numpy(np.zeros((1, 24, 3))).float().to(solver.device)
    tpose[:, 0] = matrix_to_axis_angle(solver.viz_base_R[None])[0]
    to_viz = {"cano-pose": can_pose, "t-pose": tpose, "da-pose": dapose}
    if data_provider is not None:
        to_viz["first-frame"] = pose_list[0][None]

    for name, pose in to_viz.items():
        print(f"Viz novel {name}...")
        # if export_mesh_flag:
        #     from lib_marchingcubes.gaumesh_utils import MeshExtractor
        #     # also extract a mesh
        #     mesh = solver.extract_mesh(model, pose)
        #     mesh.export(f"{viz_dir}/mc_{name}.obj", "obj")

        # # for making figures, the rotation is in another way
        # viz_spinning_self_rotate(
        #     model,
        #     solver.viz_base_R.detach(),
        #     pose,
        #     global_trans_list[0][None],
        #     H,
        #     W,
        #     K_list[0],
        #     f"{viz_dir}/{name}_selfrotate.gif",
        #     time_index=None,  # if set to None and use t, the add_bone will hand this
        #     n_spinning=n_spinning,
        #     active_sph_order=model.max_sph_order,
        # )
        viz_spinning(
            model,
            pose,
            global_trans_list[0][None],
            H,
            W,
            K_list[0],
            f"{viz_dir}/{name}.gif",
            time_index=None,  # if set to None and use t, the add_bone will hand this
            n_spinning=n_spinning,
            active_sph_order=model.max_sph_order,
            bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
        )

    # viz novel pose dynamic spinning
    print("Viz novel seq...")
    novel_pose_names = [
        f[:-4] for f in os.listdir(novel_pose_dir) if f.endswith(".npy")
    ]
    seq_viz_todo = {}
    for name in novel_pose_names:
        novel_pose_fn = osp.join(novel_pose_dir, f"{name}.npy")
        novel_poses = np.load(novel_pose_fn, allow_pickle=True)
        novel_poses = novel_poses[::novel_skip]
        N_frames = len(novel_poses)
        novel_poses = torch.from_numpy(novel_poses).float().to(solver.device)
        novel_poses = novel_poses.reshape(N_frames, 24, 3)

        seq_viz_todo[name] = (novel_poses, N_frames)
    if data_provider is not None:
        seq_viz_todo["training"] = [pose_list, len(pose_list)]

    for name, (novel_poses, N_frames) in seq_viz_todo.items():
        base_R = solver.viz_base_R.detach().cpu().numpy()
        viz_frames = []
        K = K_list[0]
        for vid in range(N_frames):
            pose = novel_poses[vid][None]
            # pose = novel_poses[0][None] # debug
            rotation = euler2mat(2 * np.pi * vid / N_frames, 0.0, 0.0, "syxz")
            rotation = torch.from_numpy(rotation @ base_R).float().to(solver.device)
            pose[:, 0] = matrix_to_axis_angle(rotation[None])[0]
            trans = global_trans_list[0][None]
            mu, fr, s, o, sph, _ = model(
                pose,
                trans,
                # not pass in {}, so t is auto none
                additional_dict={},
                active_sph_order=active_sph_order,
            )
            if model_mask is not None:
                assert len(model_mask) == mu.shape[1]
                mu = mu[:, model_mask.bool()]
                fr = fr[:, model_mask.bool()]
                s = s[:, model_mask.bool()]
                o = o[:, model_mask.bool()]
                sph = sph[:, model_mask.bool()]
            render_pkg = render_cam_pcl(
                mu[0],
                fr[0],
                s[0],
                o[0],
                sph[0],
                H,
                W,
                K,
                False,
                active_sph_order,
                bg_color=getattr(solver, "DEFAULT_BG", [1.0, 1.0, 1.0]),
                # bg_color=[1.0, 1.0, 1.0],  # ! use white bg for viz
            )
            viz_frame = (
                torch.clamp(render_pkg["rgb"], 0.0, 1.0)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            viz_frame = (viz_frame * 255).astype(np.uint8)
            viz_frames.append(viz_frame)
        imageio.mimsave(f"{viz_dir}/novel_pose_{name}.gif", viz_frames)
    return


def viz_dog_spin(
    model, pose, trans, H, W, K, save_path, n_spinning=10, device="cuda:0"
):
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    novel_view_pitch = 0.15  # or 0

    viz_frames = []
    angles = np.linspace(0.65, 0.90, n_spinning)
    angles = np.concatenate([angles, angles[::-1]])

    for angle in angles:
        rotation = euler2mat(2 * np.pi * angle, novel_view_pitch, 0.0, "syxz")
        rotation = torch.from_numpy(rotation @ BASE_R).float().to(device)
        pose[:, :3] = matrix_to_axis_angle(rotation[None])[0]

        mu, fr, s, o, sph, additional_ret = model(
            pose, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )

        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)

    imageio.mimsave(save_path, viz_frames, fps=30)

    return

def viz_dog_spin2(model, pose, trans, H, W, K, save_path, n_spinning=20):

    device = trans.device
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    trans = trans.detach().clone()

    viz_frames = []
    for vid in range(n_spinning):
        rotation = euler2mat(
            2 * np.pi * vid / n_spinning, 0.0, 0.0, "syxz"
        )
        rotation = torch.from_numpy(rotation @ BASE_R).float().to(device)
        pose[:, :3] = matrix_to_axis_angle(rotation[None])[0]
        mu, fr, s, o, sph, additional_ret = model(pose, trans, active_sph_order=model.max_sph_order)
        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )
        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)
    imageio.mimsave(save_path, viz_frames)
    return


def viz_dog_animation(
    model, animation, limb, trans, H, W, K, save_path, fps=24, device="cuda:0"
):
    yaw = 0.65
    novel_view_pitch = 0.15
    BASE_R = np.asarray(euler2mat(np.pi / 2.0, 0, np.pi, "rxyz"))
    rotation = euler2mat(2 * np.pi * yaw, novel_view_pitch, 0.0, "syxz")
    rotation = torch.from_numpy(rotation @ BASE_R).float().to(device)
    orient = matrix_to_axis_angle(rotation[None])[0]

    viz_frames = []
    for pose in animation:
        pose = pose.reshape(1, -1)
        pose = torch.cat([pose, limb], dim=1)

        mu, fr, s, o, sph, additional_ret = model(
            pose, trans, {}, active_sph_order=model.max_sph_order
        )

        render_pkg = render_cam_pcl(
            mu[0],
            fr[0],
            s[0],
            o[0],
            sph[0],
            H,
            W,
            K,
            False,
            model.max_sph_order,
            bg_color=[1.0, 1.0, 1.0],
        )

        viz_frame = (
            torch.clamp(render_pkg["rgb"], 0.0, 1.0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        viz_frame = (viz_frame * 255).astype(np.uint8)
        viz_frames.append(viz_frame)

    imageio.mimsave(save_path, viz_frames, fps=fps)


@torch.no_grad()
def viz_dog_all(solver, data_provider, model=None, ckpt_dir=None, viz_name=""):
    if model is None:
        model = solver.load_saved_model(ckpt_dir)
    model.eval()
    viz_dir = osp.join(solver.log_dir, f"{viz_name}_dog_viz")
    os.makedirs(viz_dir, exist_ok=True)

    viz_pose = (
        torch.cat([data_provider.pose_base_list, data_provider.pose_rest_list], 1)
        .detach()
        .clone()
    )
    viz_pose = torch.mean(viz_pose, dim=0, keepdim=True)   # use mean pose for viz  
    limb = viz_pose[:, -7:]                                
    pose = viz_pose[:, :-7].reshape(-1, 35, 3)
    pose[:, :-3] = 0  # exclude ears and mouth poses

    viz_pose = torch.concat([pose.reshape(1, -1), limb], dim=1)
    viz_trans = torch.tensor([[0.0, -0.3, 25.0]], device="cuda:0")

    viz_dog_spin(
        model.to("cuda"),
        viz_pose,
        viz_trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_path=osp.join(viz_dir, "spin.gif"),
        n_spinning=42,
    )

    viz_dog_spin2(
        model.to("cuda"),
        viz_pose,
        viz_trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_path=osp.join(viz_dir, "spin2.gif"),
        n_spinning=20,
    )

    ######################################################################
    # Dataset pose seq
    viz_pose = (
        torch.cat([data_provider.pose_base_list, data_provider.pose_rest_list], 1)
        .detach()
        .clone()
    )
    viz_pose = torch.mean(viz_pose, dim=0, keepdim=True)
    pose = viz_pose[:, :-7].reshape(-1, 35, 3)
    limb = viz_pose[:, -7:]

    # Animation
    aroot = osp.join(osp.dirname(__file__), "novel_poses/husky")
    window = list(range(350, 440))  # Run
    trans = torch.tensor([[0.3, -0.3, 25.0]], device="cuda:0")
    files = [f"{aroot}/{i:04d}.npz" for i in window]
    pose_list = [dict(np.load(file))["pred_pose"] for file in files]
    pose_list = np.concatenate(pose_list)
    animation = matrix_to_axis_angle(torch.from_numpy(pose_list)).to(solver.device)
    animation[:, [32, 33, 34]] = pose[:, [32, 33, 34]] 

    viz_dog_animation(
        model.to("cuda"),
        animation,
        limb,
        trans,
        data_provider.H,
        data_provider.W,
        data_provider.K_list[0],
        save_path=osp.join(viz_dir, "animation.gif"),
        fps=12,
    )
    return
