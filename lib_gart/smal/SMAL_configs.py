

import numpy as np
import os
import sys


SMAL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'smal_data')

# we replace the old SMAL model by a more dog specific model (see BARC cvpr 2022 paper)
# our model has several differences compared to the original SMAL model, some of them are:
#   - the PCA shape space is recalculated (from partially new data and weighted)  
#   - coefficients for limb length changes are allowed (similar to WLDO, we did borrow some of their code)
#   - all dogs have a core of approximately the same length
#   - dogs are centered in their root joint (which is close to the tail base)
#       -> like this the root rotations is always around this joint AND (0, 0, 0)
#       -> before this it would happen that the animal 'slips' from the image middle to the side when rotating it. Now
#          'trans' also defines the center of the rotation
#   - we correct the back joint locations such that all those joints are more aligned

# logscale_part_list = ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'] 
# logscale_part_list = ['front_legs_l', 'front_legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l', 'back_legs_l', 'back_legs_f'] 

SMAL_MODEL_CONFIG = {
  'barc': {
    'smal_model_type': 'barc',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'my_smpl_SMBLD_nbj_v3.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'my_smpl_data_SMBLD_v3.pkl'),        
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'my_smpl_data_SMBLD_v3.pkl'),
    'logscale_part_list': ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'],
  },
  '39dogs_diffsize': {
    'smal_model_type': '39dogs_diffsize',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_00791_nadine_Jr_4_dog.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_00791_nadine_Jr_4_dog.pkl'),       
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_00791_nadine_Jr_4_dog.pkl'),
    'logscale_part_list': ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'],
  },
  '39dogs_norm': {
    'smal_model_type': '39dogs_norm',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_39dogsnorm_Jr_4_dog.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_Jr_4_dog.pkl'),       
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_Jr_4_dog.pkl'), 
    'logscale_part_list': ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'],
  },
  '39dogs_norm_9ll': {    # 9 limb length parameters
    'smal_model_type': '39dogs_norm_9ll',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_39dogsnorm_Jr_4_dog.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_Jr_4_dog.pkl'),       
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_Jr_4_dog.pkl'), 
    'logscale_part_list': ['front_legs_l', 'front_legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l', 'back_legs_l', 'back_legs_f'],
  },
  '39dogs_norm_newv2': {  # front and back legs of equal lengths
    'smal_model_type': '39dogs_norm_newv2',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_39dogsnorm_newv2_dog.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_newv2_dog.pkl'),       
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_newv2_dog.pkl'), 
    'logscale_part_list': ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'],
  },
  '39dogs_norm_newv3': {  # pca on dame AND different front and back legs lengths
    'smal_model_type': '39dogs_norm_newv3',
    'smal_model_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_39dogsnorm_newv3_dog.pkl'),
    'smal_model_data_path': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_newv3_dog.pkl'),       
    'unity_smal_shape_prior_dogs': os.path.join(SMAL_DATA_DIR, 'new_dog_models', 'my_smpl_data_39dogsnorm_newv3_dog.pkl'), 
    'logscale_part_list': ['legs_l', 'legs_f', 'tail_l', 'tail_f', 'ears_y', 'ears_l', 'head_l'],
  },
}


SYMMETRY_INDS_FILE = os.path.join(SMAL_DATA_DIR, 'symmetry_inds.json')

mean_dog_bone_lengths_txt = os.path.join(SMAL_DATA_DIR, 'mean_dog_bone_lengths.txt')

# some vertex indices, (from silvia zuffi´s code, create_projected_images_cats.py)
KEY_VIDS = np.array(([1068, 1080, 1029, 1226],  # left eye
       [2660, 3030, 2675, 3038],                # right eye
       [910],                                   # mouth low
       [360, 1203, 1235, 1230],                 # front left leg, low
       [3188, 3156, 2327, 3183],                # front right leg, low
       [1976, 1974, 1980,  856],                # back left leg, low
       [3854, 2820, 3852, 3858],                # back right leg, low
       [452, 1811],                             # tail start
       [416, 235, 182],                         # front left leg, top
       [2156, 2382, 2203],                      # front right leg, top
       [829],                                   # back left leg, top
       [2793],                                  # back right leg, top
       [60, 114, 186,  59],                     # throat, close to base of neck
       [2091, 2037, 2036, 2160],                # withers (a bit lower than in reality)
       [384,  799, 1169,  431],                 # front left leg, middle
       [2351, 2763, 2397, 3127],                # front right leg, middle
       [221, 104],                              # back left leg, middle
       [2754, 2192],                            # back right leg, middle
       [191, 1158, 3116, 2165],                 # neck
       [28],                                    # Tail tip
       [542],                                   # Left Ear
       [2507],                                  # Right Ear
       [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762], # nose tip
       [0, 464, 465, 726, 1824, 2429, 2430, 2690]), dtype=object) # half tail

# the following vertices are used for visibility only: if one of the vertices is visible, 
# then we assume that the joint is visible!  There is some noise, but we don't care, as this is 
# for generation of the synthetic dataset only    
KEY_VIDS_VISIBILITY_ONLY = np.array(([1068, 1080, 1029, 1226, 645], # left eye
       [2660, 3030, 2675, 3038, 2567],                              # right eye
       [910, 11, 5],                                                # mouth low
       [360, 1203, 1235, 1230, 298, 408, 303, 293, 384],            # front left leg, low
       [3188, 3156, 2327, 3183, 2261, 2271, 2573, 2265],            # front right leg, low
       [1976, 1974, 1980,  856, 559, 851, 556],                     # back left leg, low
       [3854, 2820, 3852, 3858, 2524, 2522, 2815, 2072],            # back right leg, low
       [452, 1811, 63, 194, 52, 370, 64],                           # tail start
       [416, 235, 182, 440, 8, 80, 73, 112],                        # front left leg, top
       [2156, 2382, 2203, 2050, 2052, 2406, 3],                     # front right leg, top
       [829, 219, 218, 173, 17, 7, 279],                            # back left leg, top
       [2793, 582, 140, 87, 2188, 2147, 2063],                      # back right leg, top
       [60, 114, 186,  59, 878, 130, 189, 45],                      # throat, close to base of neck
       [2091, 2037, 2036, 2160, 190, 2164],                         # withers (a bit lower than in reality)
       [384,  799, 1169,  431, 321, 314, 437, 310, 323],            # front left leg, middle
       [2351, 2763, 2397, 3127, 2278, 2285, 2282, 2275, 2359],      # front right leg, middle
       [221, 104, 105, 97, 103],                                    # back left leg, middle
       [2754, 2192, 2080, 2251, 2075, 2074],                        # back right leg, middle
       [191, 1158, 3116, 2165, 154, 653, 133, 339],                 # neck
       [28, 474, 475, 731, 24],                                     # Tail tip
       [542, 147, 509, 200, 522],                                   # Left Ear
       [2507,2174, 2122, 2126, 2474],                               # Right Ear
       [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762],      # nose tip
       [0, 464, 465, 726, 1824, 2429, 2430, 2690]), dtype=object)   # half tail

# Keypoint indices for 3d sketchfab evaluation
SMAL_KEYPOINT_NAMES_FOR_3D_EVAL = ['right_front_paw','right_front_elbow','right_back_paw','right_back_hock','right_ear_top','right_ear_bottom','right_eye', \
                                  'left_front_paw','left_front_elbow','left_back_paw','left_back_hock','left_ear_top','left_ear_bottom','left_eye', \
                                  'nose','tail_start','tail_end']
SMAL_KEYPOINT_INDICES_FOR_3D_EVAL = [2577,	2361,	2820,	2085,	2125,	2453,	2668,	613,	394,	855,	786,	149,	486,	1079,	1845,	1820,	28]
SMAL_KEYPOINT_WHICHTOUSE_FOR_3D_EVAL = [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0]    # [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]




# see: https://github.com/benjiebob/SMALify/blob/master/config.py
# JOINT DEFINITIONS - based on SMAL joints and additional {eyes, ear tips, chin and nose}
TORSO_JOINTS = [2, 5, 8, 11, 12, 23]
CANONICAL_MODEL_JOINTS = [
  10, 9, 8, # upper_left [paw, middle, top]
  20, 19, 18, # lower_left [paw, middle, top]
  14, 13, 12, # upper_right [paw, middle, top]
  24, 23, 22, # lower_right [paw, middle, top]
  25, 31, # tail [start, end]
  33, 34, # ear base [left, right]
  35, 36, # nose, chin
  38, 37, # ear tip [left, right]
  39, 40, # eyes [left, right]
  6, 11, # withers, throat (throat is inaccurate and withers also)
  28] # tail middle
  # old:   15, 15, # withers, throat (TODO: Labelled same as throat for now), throat 

CANONICAL_MODEL_JOINTS_REFINED = [
  41, 9, 8, # upper_left [paw, middle, top]
  43, 19, 18, # lower_left [paw, middle, top]
  42, 13, 12, # upper_right [paw, middle, top]
  44, 23, 22, # lower_right [paw, middle, top]
  25, 31, # tail [start, end]
  33, 34, # ear base [left, right]
  35, 36, # nose, chin
  38, 37, # ear tip [left, right]
  39, 40, # eyes [left, right]
  46, 45, # withers, throat
  28] # tail middle

# the following list gives the indices of the KEY_VIDS_JOINTS that must be taken in order 
# to judge if the CANONICAL_MODEL_JOINTS are visible - those are all approximations!
CMJ_VISIBILITY_IN_KEY_VIDS = [
    3, 14, 8,       # left front leg
    5, 16, 10,      # left rear leg
    4, 15, 9,       # right front leg
    6, 17, 11,      # right rear leg
    7, 19,          # tail front, tail back
    20, 21,         # ear base (but can not be found in blue, se we take the tip)
    2, 2,           # mouth  (was: 22, 2)
    20, 21,         # ear tips
    1, 0,           # eyes
    18,             # withers, not sure where this point is
    12,             # throat
    23,             # mid tail
    ]

# define which bone lengths are used as input to the 2d-to-3d network
IDXS_BONES_NO_REDUNDANCY = [6,7,8,9,16,17,18,19,32,1,2,3,4,5,14,15,24,25,26,27,28,29,30,31]
# load bone lengths of the mean dog (already filtered)
mean_dog_bone_lengths = []
with open(mean_dog_bone_lengths_txt, 'r') as f:
    for line in f:
       mean_dog_bone_lengths.append(float(line.split('\n')[0]))
MEAN_DOG_BONE_LENGTHS_NO_RED = np.asarray(mean_dog_bone_lengths)[IDXS_BONES_NO_REDUNDANCY]        # (24, )

# Body part segmentation:
#   the body can be segmented based on the bones and for the new dog model also based on the new shapedirs
#   axis_horizontal = self.shapedirs[2, :].reshape((-1, 3))[:, 0]
#   all_indices =  np.arange(3889)
#   tail_indices = all_indices[axis_horizontal.detach().cpu().numpy() < 0.0]
VERTEX_IDS_TAIL = [   0,    4,    9,   10,   24,   25,   28,  453,  454,  456,  457,
        458,  459,  460,  461,  462,  463,  464,  465,  466,  467,  468,
        469,  470,  471,  472,  473,  474,  475,  724,  725,  726,  727,
        728,  729,  730,  731,  813,  975,  976,  977, 1109, 1110, 1111,
       1811, 1813, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827,
       1828, 1835, 1836, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967,
       1968, 1969, 2418, 2419, 2421, 2422, 2423, 2424, 2425, 2426, 2427,
       2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438,
       2439, 2440, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2777,
       3067, 3068, 3069, 3842, 3843, 3844, 3845, 3846, 3847]

# same as in https://github.com/benjiebob/WLDO/blob/master/global_utils/config.py
EVAL_KEYPOINTS = [
  0, 1, 2, # left front
  3, 4, 5, # left rear
  6, 7, 8, # right front
  9, 10, 11, # right rear
  12, 13, # tail start -> end
  14, 15, # left ear, right ear
  16, 17, # nose, chin
  18, 19] # left tip, right tip

KEYPOINT_GROUPS = {
  'legs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # legs
  'tail': [12, 13], # tail
  'ears': [14, 15, 18, 19], # ears
  'face': [16, 17] # face
}


