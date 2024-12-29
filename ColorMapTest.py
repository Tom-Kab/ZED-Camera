import pyzed.sl as sl
import numpy as np
import cv2
import time

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
    output_folder = "depth_images"
    
    # Create the output folder if it doesn't exist
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Capturing depth photos for 10 seconds...")

    try:
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10:  # Run for 10 seconds
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
                
                # Save the depth image
                filename = os.path.join(output_folder, f"depth_{frame_count:03d}.png")
                cv2.imwrite(filename, depth_colored)

                print(f"Saved: {filename}")
                
                frame_count += 1
                
            # Wait for 0.5 seconds
            time.sleep(0.5)

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("Capture complete. Resources released.")

if __name__ == "__main__":
    main()
