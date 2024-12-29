import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create initialization parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Or QUALITY, ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}: {}".format(err, sl.error_code_to_string(err)))
        exit()

    previous_depth_maps = []
    FILTER_LENGTH = 5
    confidence_threshold = 50
    kernel_size = 3
    global_min_depth = None
    global_max_depth = None

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            depth_map = sl.Mat()
            confidence_map = sl.Mat()

            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)

            depth_array = depth_map.get_data()
            confidence_array = confidence_map.get_data()

            # Handle invalid values (Inf and NaN)
            depth_array[np.isinf(depth_array)] = 0
            depth_array[np.isnan(depth_array)] = 0

            # Confidence Thresholding
            depth_array[confidence_array < confidence_threshold] = 0

            # Temporal Filtering
            previous_depth_maps.append(depth_array.copy())
            if len(previous_depth_maps) > FILTER_LENGTH:
                previous_depth_maps.pop(0)

            if previous_depth_maps:
                filtered_depth = np.mean(np.stack(previous_depth_maps), axis=0)

                # Spatial Filtering (Median Blur)
                filtered_depth = cv2.medianBlur(filtered_depth.astype(np.float32), kernel_size).astype(np.float32)

                # Robust Scaling with Global Min/Max
                valid_depths = filtered_depth[filtered_depth > 0]
                if valid_depths.size > 0:
                    local_min_depth = np.percentile(valid_depths, 5)
                    local_max_depth = np.percentile(valid_depths, 95)

                    if global_min_depth is None:
                        global_min_depth = local_min_depth
                        global_max_depth = local_max_depth
                    else:
                        alpha = 0.1  # Smoothing factor
                        global_min_depth = alpha * local_min_depth + (1 - alpha) * global_min_depth
                        global_max_depth = alpha * local_max_depth + (1 - alpha) * global_max_depth

                    if global_max_depth - global_min_depth > 0:
                        gray_depth = (255 * (filtered_depth - global_min_depth) / (global_max_depth - global_min_depth)).astype(np.uint8)
                        gray_depth = np.clip(gray_depth, 0, 255)
                    else:
                        gray_depth = np.zeros(filtered_depth.shape, dtype=np.uint8)
                else:
                    gray_depth = np.zeros(filtered_depth.shape, dtype=np.uint8)

                cv2.imshow("Depth Map (Grayscale)", gray_depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()

if __name__ == "__main__":
    main()