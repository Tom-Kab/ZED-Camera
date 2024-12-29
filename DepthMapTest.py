import pyzed.sl as sl
import numpy as np
import cv2

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = 300  # Minimum distance in mm
    init_params.depth_maximum_distance = 5000  # Maximum distance in mm

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Camera.")
        return

    depth = sl.Mat()
    temp_depth = None  # For temporal smoothing
    alpha = 0.8  # Smoothing factor

    print("Press 'q' to exit.")

    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_data = depth.get_data()

                # Handle invalid depth values (replace them with a specific value, e.g., 0)
                invalid_mask = np.isnan(depth_data) | (depth_data <= 0) | (depth_data > init_params.depth_maximum_distance)
                depth_data[invalid_mask] = 0

                # Temporal smoothing
                if temp_depth is None:
                    temp_depth = depth_data.copy()
                else:
                    temp_depth = cv2.addWeighted(temp_depth, alpha, depth_data, 1 - alpha, 0)

                # Normalize and visualize
                depth_normalized = cv2.normalize(temp_depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Display the depth image
                cv2.imshow("Depth Map", depth_colored)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
