import math

import cv2
import numpy as np
import pyrealsense2 as rs
from dt_apriltags import Detector

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

tag_size = 0.08

camera_params = [603.39209604, 603.68529217, 334.68581133, 255.07189437]  ##
at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=2.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

image_center_pixel = [320, 240, 1]


def isRotationMatrix(R):
    Rt = np.transpose(R)  # 旋转矩阵R的转置
    shouldBeIdentity = np.dot(Rt, R)  # R的转置矩阵乘以R
    I = np.identity(3, dtype=R.dtype)  # 3阶单位矩阵
    n = np.linalg.norm(I - shouldBeIdentity)  # np.linalg.norm默认求二范数
    return n < 1e-6  # 目的是判断矩阵R是否正交矩阵（旋转矩阵按道理须为正交矩阵，如此其返回值理论为0）


def rotationMatrixToAngles(R):
    assert (isRotationMatrix(R))  # 判断是否是旋转矩阵（用到正交矩阵特性）

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])  # 矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)

    singular = sy < 1e-6  # 判断β是否为正负90°

    if not singular:  # β不是正负90°
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:  # β是正负90°
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)  # 当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0

    ## 转成角度
    coff = 180 / 3.1415
    x = x * coff
    y = y * coff
    z = z * coff
    return np.array([x, y, z])


def detect_pose_by_apriltag():
    detect_pose = None
    # pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # pipeline.stop()

    if not color_frame:
        print(f"未获取到有效图像")
        return detect_pose

    color_image = np.asanyarray(color_frame.get_data())

    cv2.namedWindow('QR_detect', flags=cv2.WINDOW_NORMAL |
                                       cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    cv2.imshow("QR_detect", color_image)
    cv2.waitKey(30)

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

    if len(tags) >= 1:  ## 可能没有检测到

        H = tags[0].homography
        # print(f"单应性矩阵： {H}")
        corners = tags[0].corners
        # print(f"角点：{corners}")

        tmp = np.array(image_center_pixel).reshape(3, -1)
        coord_in_tag = np.dot(np.linalg.inv(H), tmp) * tag_size / 2
        # print(f"图像中心在tag坐标系下的坐标：{coord_in_tag}")

        pose_R = tags[0].pose_R

        rotate = rotationMatrixToAngles(pose_R) * -1  ## 相机相对于二维码中心的旋转与二维码相对于相机中心的旋转是反方向的

        detect_pose = {"x": float(coord_in_tag[0]),
                       "y": float(coord_in_tag[1]),
                       "yaw": rotate[-1]  # 只关心绕z轴的旋转
                       }

    return detect_pose


if __name__ == "__main__":
    while True:
        res = detect_pose_by_apriltag()

        print(f"定位结果：{res}")
