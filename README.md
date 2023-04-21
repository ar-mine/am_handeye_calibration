# am_handeye_calibration
A ROS2 package for hand-eye calibration.

## Instruction
1. Run camera node. (Use default topic names "/camera/color/image_raw" of realsense.)
2. Run rqt_image_view as monitor.
3. Launch moveit-based robot controller.
4. Modify the tf link names.
5. `ros2 topic pub /calibrator/command std_msgs/msg/Int64 "{data: 1}" --once` for add a new frame of data.
6. `ros2 topic pub /calibrator/command std_msgs/msg/Int64 "{data: 2}" --once` for save data should be saved.

## TODO List
1. GUI interface for selecting topic names of camera_image and camera_info.
2. GUI interface for selecting link names of base_link, eef_link, camera_link, sensor_link.
3. Simplify topic sender as GUI button (add a check mechanism for not +1).
4. Add list for choosing different solver algorithm.
5. Backend optimization?