from pathlib import Path
from cv2 import INTER_NEAREST
import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from lib.utils.img_utils import get_schp_palette
from plyfile import PlyData
import os.path as osp
from lib.utils.base_utils import project
from lib.utils.blend_utils import NUM_PARTS, part_bw_map, partnames


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        if cfg.zju_human != "":
            data_root = "/".join([*data_root.split('/')[:-1], cfg.zju_human])
            human = cfg.zju_human
            ann_file = "/".join([*ann_file.split('/')[:-2], cfg.zju_human, ann_file.split('/')[-1]])

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        
        breakpoint()

        if split == 'train' or split == 'prune':
            self.view = cfg.training_view
        elif split == 'test':
            if cfg.test_all_other:
                self.view = [i for i in range(num_cams) if i not in cfg.training_view]
            else:
                self.view = test_view
        elif split == 'val':
            self.view = test_view[::4]
            # self.view = test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        self.f_intv = i_intv
        ni = cfg.num_train_frame
        if cfg.record_demo and split == 'val':
            self.view = [5]
            self.tick = 0
            ni = 500
            i_intv = 1
        if cfg.test_novel_pose or cfg.aninerf_animation:
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv
            ni = cfg.num_eval_frame

        self.ims = np.array([
            np.array(ims_data['ims'])[self.view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[self.view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(self.view)

        self.lbs_root = os.path.join(self.data_root, cfg.lbs)
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        self.nrays = cfg.N_rand

        self.error_map = None
        self.hull = [None for _ in range(cfg.num_train_frame)]

        if cfg.use_knn:
            faces, weights, joints, parents, parts = self.load_smpl()
            self.meta_smpl = {'faces': faces, 'weights': weights, 'joints': joints, 'parents': parents, 'parts': parts}


    def load_smpl(self):
        smpl_meta_root = cfg.smpl_meta
        faces = np.load(os.path.join(smpl_meta_root, 'faces.npy')).astype(np.int64)
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy')).astype(np.float32)
        parents = np.load(os.path.join(smpl_meta_root, 'parents.npy')).astype(np.int64)
        weights = np.load(os.path.join(smpl_meta_root, 'weights.npy')).astype(np.float32)
        parts = np.zeros((6890,))
        weights_max = weights.argmax(axis=-1)
        for pid in range(NUM_PARTS):
            partname = partnames[pid]
            part_bwids = part_bw_map[partname]
            for bwid in part_bwids:
                parts[weights_max == bwid] = pid

        return faces, weights, joints, parents, parts

    def init_global(self, **kwargs):
        self.error_map = np.ones((cfg.num_train_frame, len(self.view), kwargs['H'], kwargs['W']), dtype=np.float32) * 1000

    def get_hull(self, index, wbounds):
        img_path = self.ims[index]
        i = int(os.path.basename(img_path).split('_')[4]) - 1
        base_index = (index // self.num_cams) * self.num_cams
        Path(osp.join(cfg.result_dir, "hull")).mkdir(exist_ok=True, parents=True)
        if self.hull[i] is None:
            hull_path = osp.join(cfg.result_dir, 'hull', "{}.npy".format(i))
            if osp.exists(hull_path):
                self.hull[i] = np.load(hull_path).astype(np.float32)
            else:
                wpts = if_nerf_dutils.get_grid_points_with_bound(wbounds)
                flag = np.ones((*wpts.shape[:-1], 1)).astype(int)
                for vi, view in enumerate(self.view):
                    new_index = base_index + vi
                    mask_orig = self.get_mask(new_index)[1]
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.dilate(mask_orig, kernel)
                    cam_ind = self.cam_inds[new_index]

                    K = np.array(self.cams['K'][cam_ind])
                    D = np.array(self.cams['D'][cam_ind])
                    R = np.array(self.cams['R'][cam_ind])
                    T = np.array(self.cams['T'][cam_ind]) / 1000.

                    pts2d = project(wpts.reshape(-1, 3), K, np.concatenate((R, T), axis=1)).reshape(*wpts.shape[:-1], 2).astype(int)
                    coord = np.clip(pts2d, a_min=0, a_max=mask.shape[0] - 1)

                    occ = mask[coord[..., 1], coord[..., 0]]
                    flag &= (occ[..., None] > 0)
                self.hull[i] = flag.astype(np.float32)
                np.save(hull_path, flag)
                import mcubes
                import trimesh
                cube = np.pad(flag[..., 0], 10, mode='constant')
                verts, triangles = mcubes.marching_cubes(cube, 0.5)
                verts = (verts - 10) * cfg.voxel_size[0]
                verts = verts + wbounds[0]
                mesh = trimesh.Trimesh(vertices=verts, faces=triangles)
                mesh.export(os.path.join(cfg.result_dir, "hull", "mesh_{}.ply".format(i)))

        return self.hull[i]

    def load_global(self):
        if not cfg.sample_using_mse or (self.error_map is not None and self.error_map.min() < 1000):
            return
        error_cache = osp.join(cfg.result_dir, "latest_error.npy")
        if osp.exists(error_cache):
            self.error_map = np.load(error_cache)
        return

    def save_global(self):
        if not cfg.sample_using_mse or self.error_map is None:
            return
        error_cache = osp.join(cfg.result_dir, "latest_error.npy")
        np.save(error_cache, self.error_map)
        error_map_vis = self.error_map[0, 0] != 1000
        cv2.imwrite(osp.join(cfg.result_dir, "latest_error.png"), error_map_vis.astype(np.uint8) * 255)

    def update_global(self, ret, batch):
        if cfg.sample_using_mse and self.error_map is not None:
            coord = batch['coord'][0].detach().cpu().numpy()
            err = ret['error'][0].detach().cpu().numpy()
            cind = self.view.index(batch['cam_ind'][0])
            self.error_map[batch['frame_index'] // self.f_intv, cind, coord[:, 0], coord[:, 1]] = err

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, self.ims[index].replace('images', 'schp'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask_cihp', self.ims[index])[:-4] + '.png'
            msk_cihp = imageio.imread(msk_path)[..., None].repeat(3, axis=-1)
        else:
            msk_cihp = imageio.imread(msk_path)
            msk_cihp = np.array(msk_cihp)[:, :, :3].astype(np.uint8)  # H, W, 3

            palette = get_schp_palette(cfg.semantic_dim)
            sem_msk = np.zeros(msk_cihp.shape[:2], dtype=np.uint8)
            for i, rgb in enumerate(palette):
                belong = (msk_cihp - rgb).sum(axis=-1) == 0
                sem_msk[belong] = i
            msk_cihp = sem_msk

        face_msk = (msk_cihp == 2) | (msk_cihp == 10) | (msk_cihp == 13)
        larm_msk = (msk_cihp == 14)
        rarm_msk = (msk_cihp == 15)
        lleg_msk = (msk_cihp == 9) | (msk_cihp == 16)
        rleg_msk = (msk_cihp == 9) | (msk_cihp == 17)
        body_msk = (msk_cihp == 5)
        arm_msk = larm_msk | rarm_msk
        leg_msk = lleg_msk | rleg_msk

        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            # if self.human == 'my_377':
            #     border = 20
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        semantic_masks = {
            "head": face_msk,
            "larm": larm_msk,
            "rarm": rarm_msk,
            'lleg': lleg_msk,
            'rleg': rleg_msk,
            "leg": leg_msk,
            'body': body_msk,
            'arm': arm_msk,
        }

        for k in semantic_masks:
            smask = semantic_masks[k]
            smask = smask.astype(np.uint8)
            semantic_masks[k] = smask

        return msk, orig_msk, semantic_masks

    def get_normal(self, index):
        normal_path = os.path.join(self.data_root, 'normal',
                                   self.ims[index])[:-4] + '.png'
        normal = imageio.imread(normal_path) / 255.
        normal = normal * 2 - 1
        return normal

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)

        if cfg.mono_bullet:
            axis = np.array([0., 1., 0.], dtype=np.float32)
            Rrel = cv2.Rodrigues(i * axis)[0]
            wxyz = (wxyz - Th) @ Rrel.T + Th
            R = Rrel @ R
            Rh = cv2.Rodrigues(R)[0].reshape(1, 3)

        # prepare sp input of param pose
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)

        big_poses = np.zeros_like(poses).ravel()
        if cfg.tpose_geometry:
            angle = 30
            big_poses[5] = np.deg2rad(angle)
            big_poses[8] = np.deg2rad(-angle)
        else:
            big_poses = big_poses.reshape(-1, 3)
            big_poses[1] = np.array([0, 0, 7. / 180. * np.pi])
            big_poses[2] = np.array([0, 0, -7. / 180. * np.pi])
            big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
            big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, joints, parents)
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, big_A, pbw, Rh, Th

    def __getitem__(self, index):
        if self.split == 'train':
            ratio = cfg.ratio
        else:
            # __import__('ipdb').set_trace()
            ratio = cfg.eval_ratio

        if self.split == 'val' and cfg.record_demo:
            index = self.tick % len(self.ims)
            print(self.tick)
            self.tick += 1

        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk, semantic_masks = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        for k in semantic_masks:
            smask = semantic_masks[k]
            smask = cv2.resize(smask, (W, H), interpolation=cv2.INTER_NEAREST)
            semantic_masks[k] = smask

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)
        # face_msk = cv2.undistort(face_msk, K, D)
        for k in semantic_masks:
            smask = semantic_masks[k]
            smask = cv2.undistort(smask, K, D)
            semantic_masks[k] = smask

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        # face_msk = cv2.resize(face_msk, (W, H),
        #                       interpolation=cv2.INTER_NEAREST)
        for k in semantic_masks:
            smask = semantic_masks[k]
            smask = cv2.resize(smask, (W, H), interpolation=cv2.INTER_NEAREST)
            semantic_masks[k] = smask
        img_oldold = img.copy()
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # read v_shaped
        if cfg.bigpose:
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        # save_point_cloud(tpose, 'debug/tpose_{}.ply'.format(get_time()))
        tbounds = if_nerf_dutils.get_bounds(tpose)
        if cfg.bigpose:
            tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        else:
            tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))
        tbw = tbw.astype(np.float32)

        tuv = np.load(os.path.join(self.data_root, 'bigpose_uv.npy'))
        head_uv = None
        try:
            head_uv = np.load(osp.join(self.data_root, "head_uv.obj"))
        except:
            pass
        # tuvh = np.load(os.path.join(self.data_root, 'bigpose_uvh.npy'))
        # puvh = np.load(os.path.join(self.data_root, 'uvh', 'uvh_{:03d}.npy'.format(i)))
        # puvh = np.abs(puvh)

        wpts, ppts, A, big_A, pbw, Rh, Th = self.prepare_input(i)

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        if cfg.prune_using_hull:
            hull = self.get_hull(index, wbounds)
        else:
            hull = None

        thresh = None

        breakpoint()
        if cfg.train_with_coord and self.split == 'train':
            coord_path = os.path.join(
                self.data_root,
                'train_coord/frame_{:04d}_view_{:04d}.npy'.format(
                    frame_index, cam_ind))
            train_coord = np.load(coord_path, allow_pickle=True).item()
            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_coord(
                img, msk, train_coord, K, R, T, wbounds, self.nrays)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]
        elif cfg.sample_using_mse and self.split == 'train':
            if self.error_map is None:
                self.init_global(H=H, W=W)
                self.load_global()
            cind = self.view.index(cam_ind)
            error_map = self.error_map[frame_index // self.f_intv, cind]
            nonz_error_map = error_map[(error_map > 0) & (msk == 1)]
            sample_coord_len = int(nonz_error_map.shape[0] * 0.2)
            ind = np.argpartition(nonz_error_map, -sample_coord_len)[-sample_coord_len:]
            thresh = nonz_error_map[ind].min()
            error_msk = error_map >= thresh
            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m_mse(img, msk, error_msk, K, R, T, wbounds, self.nrays, self.split)
            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]
        elif (cfg.use_lpips or cfg.patch_sampling or cfg.use_ssim or cfg.use_fourier or cfg.use_tv_image) and self.split == 'train':
            img_old = img.copy()
            msk_old = msk.copy()
            if cfg.sample_focus == "" or semantic_masks[cfg.sample_focus].sum() == 0:
                ret_crop = if_nerf_dutils.crop_image_msk(img, msk, K, msk)
            else:
                smask = semantic_masks[cfg.sample_focus]
                # img, msk, K, _ = if_nerf_dutils.crop_image_msk(img, msk, K, smask)
                ret_crop = if_nerf_dutils.crop_image_msk(img, msk, K, smask)
            if ret_crop is not None:
                img, msk, K, _ = ret_crop
            else:
                pass

            if self.split == 'train':
                img, msk, K = if_nerf_dutils.random_crop_image(img, msk, K)
            H, W = img.shape[:2]
            ray_o, ray_d, near, far, mask_at_box, coord = if_nerf_dutils.get_rays_within_bounds_coord(H, W, K, R, T, wbounds)
            rgb = img[mask_at_box]
            mask_at_box = mask_at_box.reshape(-1)
            occupancy = (msk[coord[:, 1], coord[:, 0]] > 0)
        else:
            breakpoint()
            flag = cfg.prune_using_geo and os.path.exists(os.path.join(cfg.result_dir, "latest.npy"))
            nrays = self.nrays if not flag else 2 * self.nrays
            rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
                img, msk, K, R, T, wbounds, nrays, self.split)
            if cfg.erode_edge:
                orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
            occupancy = orig_msk[coord[:, 0], coord[:, 1]]
            # assert np.all(mask_at_box)

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'coord': coord,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        if cfg.train_with_normal:
            # normal is extracted from undistroted image
            normal = self.get_normal(index)
            normal = cv2.resize(normal, (W, H),
                                interpolation=cv2.INTER_NEAREST)
            normal = normal[coord[:, 0], coord[:, 1]].astype(np.float32)
            ret.update({'normal': normal})

        # blend weight
        meta = {
            'A': A,
            'big_A': big_A,
            'pbw': pbw,
            'tbw': tbw,
            'tuv': tuv,
            # 'tuvh': tuvh,
            # 'puvh': puvh,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        bw_latent_index = index // self.num_cams
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        frame_dim = np.array(latent_index / cfg.num_train_frame).astype(np.float32)
        if cfg.record_demo and self.split == 'val':
            frame_dim = np.array(float(frame_dim / 5)).astype(np.float32)
            from math import floor
            latent_index = int(floor(latent_index / 5))
            bw_latent_index = int(floor(bw_latent_index / 5))
        meta = {
            'frame_dim': frame_dim,
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        # prune using geometry
        prev_geometry = np.zeros(1)
        geometry_thresh = 0.2
        if os.path.exists(os.path.join(cfg.result_dir, "latest.npy")) and cfg.prune_using_geo:
            prev_geometry = np.load(os.path.join(cfg.result_dir, "latest.npy"))
            N = (prev_geometry > -1).sum()
            # NN = int(N * 0.15)
            NN = int(N * 0.1)
            ccube = prev_geometry.reshape(-1)
            ind = np.argpartition(ccube, -NN)[-NN:]
            geometry_thresh = ccube[ind].min()

        ret.update({
            "prev_geometry": prev_geometry,
            "geo_thresh": geometry_thresh
        })

        if thresh is not None:
            ret.update({"thresh": thresh})

        if hull is not None:
            ret.update({"hull": hull})

        if head_uv is not None:
            ret.update({"head_uv": head_uv})

        # part = self.meta_smpl['parts']
        # face = self.meta_smpl['faces']
        # import trimesh
        # mesh = trimesh.load_mesh(os.path.join('./smpl.ply'))
        # tverts = tpose
        # print(tpose.shape)
        # for pid in range(NUM_PARTS):
        #     partname = partnames[pid]
        #     face_flag = (part[face[..., 0]] == pid) | (part[face[..., 1]] == pid) | (part[face[..., 2]] == pid)
        #     part_faces = face[face_flag]
        #     part_mesh = trimesh.Trimesh(vertices=tverts, faces=part_faces)
        #     # save mesh
        #     part_mesh.export(os.path.join('debug', partname + "_tpose3.obj"))

        # breakpoint()

        sem_masks = []
        for key in partnames:
            sem_masks.append(semantic_masks[key])
        sem_masks = np.stack(sem_masks, axis=0)

        ret.update({"sem_mask": sem_masks})

        if cfg.use_knn:
            ret.update({'ppts': ppts, 'wpts': wpts, 'tpts': tpose})
            if cfg.n_coarse_knn_ref != -1:
                ret['sub_ppts'] = ppts[np.random.permutation(cfg.n_coarse_knn_ref)]
            else:
                ret['sub_ppts'] = ppts

            ret.update(self.meta_smpl)

            N, D = self.meta_smpl['weights'].shape
            P = NUM_PARTS

            pose_verts = ppts
            pose_bw = self.meta_smpl['weights']

            pts_part = self.meta_smpl['parts']

            part_pts = np.zeros((P, N, 3), dtype=np.float32)
            part_pbw = np.zeros((P, N, D), dtype=np.float32)
            lengths2 = np.zeros(P, dtype=int)
            bounds = np.zeros((P, 2, 3), dtype=np.float32)
            for pid in range(P):
                part_flag = (pts_part == pid)
                lengths2[pid] = np.count_nonzero(part_flag)
                part_pts[pid, :lengths2[pid]] = pose_verts[part_flag]
                part_pbw[pid, :lengths2[pid]] = pose_bw[part_flag]

                bounds[pid, 0] = tpose[part_flag].min(axis=0) - cfg.bbox_overlap
                bounds[pid, 1] = tpose[part_flag].max(axis=0) + cfg.bbox_overlap

            max_length = lengths2.max()
            part_pts = part_pts[:, :max_length, :]
            part_pbw = part_pbw[:, :max_length, :]

            ret.update({
                'part_pts': part_pts,
                'part_pbw': part_pbw,
                'lengths2': lengths2,
                'bounds': bounds,
            })

        return ret

    def __len__(self):
        if cfg.record_demo and self.split == 'val':
            return 1
        return len(self.ims)
