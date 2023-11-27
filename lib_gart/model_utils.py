import numpy as np
from plyfile import PlyData, PlyElement
import torch
from pytorch3d.transforms import matrix_to_quaternion
from copy import deepcopy
import trimesh
from torch.distributions.multivariate_normal import MultivariateNormal


def save_gauspl_ply(path, xyz, frame, scale, opacity, color_feat):
    # ! store in gaussian splatting activation way: opacity use sigmoid and scale use exp
    xyz = xyz[0].detach().cpu().numpy().squeeze()
    N = xyz.shape[0]
    normals = np.zeros_like(xyz)
    sph_feat = color_feat[0].reshape(N, -1, 3)
    f_dc = (
        sph_feat[:, :1]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )  # ! self._features_dc: N,1,3
    f_rest = (
        sph_feat[:, 1:]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )  # ! self._features_rest: N, 15, 3
    opacities = (
        torch.logit(opacity[0]).detach().cpu().numpy()
    )  # ! self._opacity, before comp, N,1
    scale = torch.log(scale[0]).detach().cpu().numpy()  # ! _scaling, N,3, before comp
    # rotation = self._rotation.detach().cpu().numpy() # ! self._rotation, N,4 quat
    rotation = (
        matrix_to_quaternion(frame[0]).detach().cpu().numpy()
    )  # ! self._rotation, N,4 quat

    dtype_full = [
        (attribute, "f4")
        for attribute in construct_list_of_attributes(sph_feat.shape[1] - 1)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def construct_list_of_attributes(_features_rest_D):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(1 * 3):
        l.append("f_dc_{}".format(i))
    for i in range(_features_rest_D * 3):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(3):
        l.append("scale_{}".format(i))
    for i in range(4):
        l.append("rot_{}".format(i))
    return l


def transform_mu_frame(mu, frame, T):
    if len(mu) != len(T):
        assert len(mu) == 1 and len(frame) == 1
        mu = mu.expand(len(T), -1, -1)
        frame = frame.expand(len(T), -1, -1, -1)
    R, t = T[:, :3, :3], T[:, :3, 3]
    new_frame = torch.einsum("bij, bnjk->bnik", R, frame)
    new_mu = torch.einsum("bij, bnj->bni", R, mu) + t[:, None]
    return new_mu, new_frame


def mesh_coordinate(xyz, mesh):
    closest_pts, dist, face_ind = trimesh.proximity.closest_point(mesh, xyz)
    triangles = get_triangles(mesh.vertices, mesh.faces, face_ind)
    barycentric_w = trimesh.triangles.points_to_barycentric(
        triangles, xyz, method="cramer"
    )
    return face_ind, barycentric_w


def get_triangles(base_vtx, base_face, face_ind):
    # get the triangles for barycentric weight
    face_selected = deepcopy(base_face)[face_ind]  # N,3
    i1, i2, i3 = face_selected[:, 0], face_selected[:, 1], face_selected[:, 2]  # N,3
    _vtx = deepcopy(base_vtx)
    v1, v2, v3 = (
        _vtx[i1][:, np.newaxis, :],
        _vtx[i2][:, np.newaxis, :],
        _vtx[i3][:, np.newaxis, :],
    )
    triangles = np.concatenate([v1, v2, v3], axis=1)
    return triangles  # N,3,3


def sph_order2nfeat(order):
    return (order + 1) ** 2


def get_predefined_human_rest_pose(pose_type):
    print(f"Using predefined pose: {pose_type}")
    body_pose_t = torch.zeros((1, 69))
    if pose_type.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif pose_type.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    elif pose_type.lower() == "t_pose":
        pass
    else:
        raise ValueError("Unknown cano_pose: {}".format(pose_type))
    return body_pose_t.reshape(23, 3)


def get_predefined_dog_rest_pose(pose_type):
    if pose_type == "standing" or pose_type == "t_pose" or pose_type == "zero_pose":
        pose = torch.zeros(34 * 3 + 7)
    elif pose_type == "da_pose":
        pose = torch.zeros(35, 3)
        pose[7, 0] = np.pi / 8
        pose[11, 0] = -np.pi / 8
        pose[17, 0] = np.pi / 8
        pose[21, 0] = -np.pi / 8
        pose = torch.concatenate([pose[1:].reshape(-1), torch.zeros(7)], -1)
    else:
        raise NotImplementedError()
    return pose  # 34*3+7


def sample_pcl_from_gaussian(mu, frame, scale, k=6):
    # TODO: have to have a factor
    n = len(mu) * k
    device = mu.device
    _mean, _cov = torch.zeros(3), torch.eye(3)
    mvn = MultivariateNormal(_mean.to(device), _cov.to(device))
    samples = mvn.sample((n,))
    densities = torch.exp(mvn.log_prob(samples))
    samples = samples.reshape(len(mu), k, 3)
    densities = densities.reshape(len(mu), k)
    world_coordinate_samples = (
        torch.einsum("nij,nkj->nki", frame, samples) * scale[:, None]
    )
    world_coordinate_samples = world_coordinate_samples + mu[:, None, :]
    # # debug
    # np.savetxt("../../debug/samples.xyz", torch.cat([world_coordinate_samples, densities[...,None]],-1).reshape(-1,4).detach().cpu().numpy(), fmt="%.6f")
    # np.savetxt("../../debug/original.xyz", mu.detach().cpu().numpy(), fmt="%.6f")
    return world_coordinate_samples, densities


def dep_back_proj(K, d):
    # K: 3,3
    # d: 1,H,W
    # return: 3,H,W
    H, W = d.shape[-2:]
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    y, x = y.to(d.device), x.to(d.device)
    xy = torch.stack([x, y, torch.ones_like(x)], dim=0).float()  # 3,H,W
    pcl = torch.einsum("ij, jhw->ihw", torch.inverse(K), xy) * d  # 3,H,W
    return pcl
