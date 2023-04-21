import os

import numpy as np
import cv2
from cv2 import aruco


class ArucoBoardDetector:
    def __init__(self):
        self.charuco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.charuco_board = aruco.CharucoBoard((5, 7), 0.04, 0.02, self.charuco_dictionary)

    def create_board(self, save: str = './'):
        # 生成Charuco棋盘的图片
        img_charuco = self.charuco_board.generateImage((500, 600), 10, 1)

        # 保存Charuco棋盘的图片
        cv2.imwrite(os.path.join(save, 'charuco_board.jpg'), img_charuco)


    def detect_board(self, image, camera_matrix, dist_coeffs, show_markers: bool = True, show_corners: bool = True):
        pose_valid = False
        image_copy = image.copy()
        marker_corners, marker_ids, rejected = aruco.detectMarkers(image, self.charuco_board.getDictionary())
        if marker_ids is not None:
            image_copy = image.copy()
            if show_markers:
                aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)

            charuco_num, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, image_copy, self.charuco_board)

            if charuco_num > 0:
                if show_corners:
                    aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids, (255, 0, 0))
                valid, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, camera_matrix, dist_coeffs, None, None)
                if valid:
                    cv2.drawFrameAxes(image_copy, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                    pose_valid = True
                    transform = np.eye(4)
                    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
                    transform[:3, 3] = tvec[:, 0]
                    return pose_valid, image_copy, transform

        return pose_valid, image_copy


if __name__ == "__main__":
    detector = ArucoBoardDetector()
    detector.create_board()

    image = cv2.imread("./charuco_board.jpg")
    camera_matrix = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    pose_valid, image_marked, transform = detector.detect_board(image, camera_matrix, dist_coeffs, show_markers=False, show_corners=False)
    print(transform)
    cv2.imshow("out", image_marked)
    cv2.waitKey(0)



