import pickle
import numpy as np
import transforms3d as t3d
import cv2

AVAILABLE_ALGORITHMS = {
        'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def rotation_inv(transforms):
    ret = []
    for transform in transforms:
        ret.append(transform.T)
    return ret


def translation_inv(transforms):
    ret = []
    for transform in transforms:
        ret.append(-transform)
    return ret


if __name__ == "__main__":
    with open("./calibration.pkl", 'rb') as f:
        data = pickle.load(f)

    board_R_list = data['board_R_list']
    board_T_list = data['board_T_list']
    eef_R_list = data['eef_R_list']
    eef_T_list = data['eef_T_list']
    image_list = data['images']

    standard_transform_array = np.array([0.0513706, -0.045675, 0.0331922, 0.699842, 0.0103786, 0.714221, -0.00154157])
    standard_transform = np.eye(4)
    standard_transform[:3, :3] = t3d.quaternions.quat2mat([-0.00154157, 0.699842, 0.0103786, 0.714221])
    standard_transform[:3, 3] = [0.0513706, -0.045675, 0.0331922]

    result_transform = np.array([0.460, -0.184, 0.031, 0.025, 0.999, -0.023, -0.019])

    num = len(board_R_list)

    solved_transform_R, solved_transform_T = cv2.calibrateHandEye(eef_R_list, eef_T_list, board_R_list, board_T_list, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
    solved_transform = np.eye(4)
    solved_transform[:3, :3] = solved_transform_R
    solved_transform[:3, 3] = solved_transform_T[:, 0]
    solved_transform_quat = t3d.quaternions.mat2quat(solved_transform_R)
    print([solved_transform_T[:, 0],
           *solved_transform_quat[1:], solved_transform_quat[0]])

    composed_transform_list = []
    for i in range(num):
        board_transform = np.eye(4)
        board_transform[:3, :3] = board_R_list[i]
        board_transform[:3, 3] = board_T_list[i]

        eef_transform = np.eye(4)
        eef_transform[:3, :3] = eef_R_list[i]
        eef_transform[:3, 3] = eef_T_list[i]

        composed_transform = eef_transform @ solved_transform @ board_transform
        composed_transform_quat = t3d.quaternions.mat2quat(composed_transform[:3, :3])
        composed_transform = np.array([*composed_transform[:3, 3],
                                       *composed_transform_quat[1:], composed_transform_quat[0]])

        composed_transform_list.append(composed_transform)

    composed_transform_list = np.array(composed_transform_list)
    print(np.mean(np.std(composed_transform_list, axis=0)))
    # 0.003592334060016828, 0.024472237352299784
    # print(cv2.calibrateHandEye(rotation_inv(eef_R_list), translation_inv(eef_T_list), rotation_inv(board_R_list), translation_inv(board_T_list)))
