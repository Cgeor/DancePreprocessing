import os
import numpy as np
import json
'''This script outputs landmarks and setup json files readable by multiview calib for each frame of the sequence.
It compiles the information of each camera and ensure that the detections are properly assigned by reading the anchor_dict.'''

def get_frame_info(frame_path):
    "Returns two lists of arrrays of detections and confidence for each detected person in the frame"
    pos = []
    conf = []
    with open(frame_path) as jsonfile:
        data = json.load(jsonfile)

    for ppl in data['people']:
        p0 = np.array(ppl['pose_keypoints_2d'])
        p0_xy = np.zeros(50)
        p0_c = p0[2::3]
        p0_xy[0::2] = p0[0::3]
        p0_xy[1::2] = p0[1::3]
        pos.append(p0_xy)
        conf.append(p0_c)

    return pos, conf


def swap_distance_check(last_pos, curr_pos):
    '''Check if the position in the array needs to be flipped. The condition is that the minimum distance has to be crossed
    for the array to be flipped.'''
    assert len(last_pos) == 2
    assert len(curr_pos) == 2
    d00 = pose_distance(last_pos[0], curr_pos[0])
    d10 = pose_distance(last_pos[1], curr_pos[0])
    d01 = pose_distance(last_pos[0], curr_pos[1])
    d11 = pose_distance(last_pos[1], curr_pos[1])
    dist_array = np.array([d00, d11, d01, d10])

    if np.argmin(dist_array) in (0, 1):
        return False
    else:
        return True


def process_frame_info(pos, conf, last_pos):
    '''Do some preprocessing of the data before assigning poses. We keep only the two first that have the highest confidence
    (not ideal but ok for now). '''
    scores = np.average(conf, -1)  # sort by average confidence in descending order
    scores_sorted, pos_sorted = (list(t[::-1]) for t in zip(*sorted(zip(scores, pos))))

    if len(pos) > 2:
        pos_sorted = pos_sorted[0:2] # maximum two people in the scene so we cut off the rest
        if scores_sorted[1] < 0.2 * scores_sorted[0]: # if the second score is too low its probably a false detection
            pos_sorted[1] = np.full(50, np.inf)
            conf[1] = np.zeros(25)

    if len(pos) == 1:
        pos_sorted.append(np.full(50, np.inf)) # append a zero array if only one person is visible in the scene
        conf.append(np.zeros(25))

    if last_pos is not None:
        if len(last_pos) == 2 and len(pos_sorted) == 2:
            if swap_distance_check(last_pos, pos_sorted):
                pos_sorted[0], pos_sorted[1] = pos_sorted[1], pos_sorted[0]
                conf[0], conf[1] = conf[1], conf[0]
                # if there are two elements and the distance from the sorted list is inferior, swap them

    return pos_sorted, conf


def pose_distance(pose, last_pos):
    '''Compute the distance between the current and last pose, returns inf if any of the pose has infinite norm or
    if there are no corresponding points between the two poses. The distance is actually divided by the size of the array
     squared so that poses with a lot of the same points (which are more likely to be the same person) are more likely paired'''

    if np.linalg.norm(pose) == np.inf or np.linalg.norm(last_pos) == np.inf:
        return np.inf
    else:
        r = abs(last_pos - pose) * (pose * last_pos != 0.0)
        if not np.any(r):
            return np.inf
        else:
            r_nonzero = r[r != 0.0]
            return np.average(r_nonzero / (r_nonzero.size * r_nonzero.size))


def make_pairs(list):
    '''Make pairs of camera for the setup file. The number of cameras varies with the frame and subject.'''
    n = len(list)
    pairs = []
    for i in range(n-1):
        pairs.append([list[i], list[i+1]])

    return pairs


# note : we have to assume no false detection for first frame
# in 6.3, the woman (p1) is not visible in the first frame. Her position will be set to 0 in process_frame info so
# we mark her position as left
 # p1_initial_placement = ['left', 'right', 'right', 'left', 'left']

data_path = '/home/costa/data/seq7_cut/'
''' 
# this was used for seq 1
cam0_path = data_path + 'data4.2'
cam1_path = data_path + 'data6.0'
cam2_path = data_path + 'data6.1'
cam3_path = data_path + 'data6.3_blur'
cam4_path = data_path + 'dataG2'
cam_paths = [cam0_path, cam1_path, cam2_path, cam3_path, cam4_path]
cam_nbr = len(cam_paths)
'''
# path of each cameras
cam0_path = data_path + 'data4.0'
cam1_path = data_path + 'data4.2'
cam2_path = data_path + 'data6.0'
cam3_path = data_path + 'data6.1'
cam4_path = data_path + 'data6.2'
cam5_path = data_path + 'data6.3'
cam6_path = data_path + 'dataG2'
cam_paths = [cam0_path, cam1_path, cam2_path, cam3_path, cam4_path, cam5_path, cam6_path]
cam_nbr = len(cam_paths)
cam_entries = [sorted(os.listdir(cam_path)) for cam_path in cam_paths]
frame_nbr = len(cam_entries[0])
last_pos = [None] * cam_nbr

# load the "anchors", annotations for whether the person labelled as p1 (first on the list) is on the left or right of
# the image. The distance matching does not work all the time due to false partial detections so some frames have to be
# annotated by hand
with open('anchor_dict.json') as json_file:
    anchor_dict = json.load(json_file)

# for each frame, load data for each camera, load and process data, and dump it in a json file in a format readable for multiview_calib
for frame in range(frame_nbr):
    dict_list = [{}, {}]
    cam_used = [[], []]
    setup_dict_list = [{}, {}]
    for cam in range(cam_nbr):
        too_close_flag = False
        pos, conf = get_frame_info(f'{cam_paths[cam]}/{cam_entries[cam][frame]}')
        pos, conf = process_frame_info(pos, conf, last_pos[cam])

        p1_mean_x = np.mean(pos[0][::2][pos[0][::2] != 0.0])
        p2_mean_x = np.mean(pos[1][::2][pos[1][::2] != 0.0])
        if abs(p1_mean_x - p2_mean_x) < 10: # average x position is less than x pixels
            too_close_flag = True
        if f'frame_{frame}' in anchor_dict:

            if (p1_mean_x < p2_mean_x and anchor_dict[f'frame_{frame}'][cam] == 'right') or (p1_mean_x > p2_mean_x and anchor_dict[f'frame_{frame}'][cam] == 'left'):
                pos[0], pos[1] = pos[1], pos[0]
                conf[0], conf[1] = conf[1], conf[0]


        for subj in range(len(pos)):
            ids = []
            landmarks = []
            subj_pos = pos[subj]
            subj_conf = conf[subj]
            cam_dict = {}
            for param_idx in range(len(subj_conf)):
                if subj_conf[param_idx] > 0.0 and subj_pos[2 * param_idx : 2 * param_idx + 2].tolist() != [0.0, 0.0]: # need to check for both conditions
                    ids.append(param_idx)
                    landmarks.append(subj_pos[2 * param_idx : 2 * param_idx + 2].tolist())

            if len(ids) > 12 and not too_close_flag:
                cam_dict["ids"] = ids
                cam_dict["landmarks"] = landmarks
                dict_list[subj][f'cam{cam}'] = cam_dict
                cam_used[subj].append(f'cam{cam}')

            setup_dict_list[subj]["views"] = cam_used[subj]
            setup_dict_list[subj]["minimal_tree"] = make_pairs(cam_used[subj])

        with open(f'output_files/subject1/setup/setup_sub1_frame_{frame}.json', 'w') as outfile:
            json.dump(setup_dict_list[0], outfile, indent=2)
        with open(f'output_files/subject2/setup/setup_sub2_frame_{frame}.json', 'w') as outfile:
            json.dump(setup_dict_list[1], outfile, indent=2)

        with open(f'output_files/subject1/landmarks/landmarks_sub1_frame_{frame}.json', 'w') as outfile:
            json.dump(dict_list[0], outfile, indent=2)
        with open(f'output_files/subject2/landmarks/landmarks_sub2_frame_{frame}.json', 'w') as outfile:
            json.dump(dict_list[1], outfile, indent=2)

        last_pos[cam] = pos
