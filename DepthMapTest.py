import pyzed.sl as sl
import numpy as np
import cv2

def main():
    # Create a ZED Camera object
    zed = sl.Camera()

    # Set up the ZED Camera configuration
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode for high precision
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter as the unit for depth measurements

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open the ZED Camera.")
        return

    # Create Mat objects to store images
    image = sl.Mat()
    depth = sl.Mat()

    print("Press 'q' to exit.")

    try:
        while True:
            # Capture images and retrieve the depth map
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)  # Retrieve the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Retrieve the depth map

                # Convert the depth map to a numpy array
                depth_data = depth.get_data()
                
                # Normalize the depth data for visualization
                depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                depth_data_colored = cv2.applyColorMap(depth_data_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                # Display the images
                cv2.imshow("Image", image.get_data())
                cv2.imshow("Depth Map", depth_data_colored)

                # Exit the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # Release resources
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
