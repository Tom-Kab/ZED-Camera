import math
import pyzed.sl as sl

# Create a ZED camera
zed = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)
  
# Capture 50 images and depth, then stop
i = 0
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()
while (i < 50):
    # Grab an image
    if (zed.grab() == sl.ERROR_CODE.SUCCESS):
        # A new image is available if grab() returns SUCCESS
        zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Retrieve depth Mat. Depth is aligned on the left image
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # Retrieve point cloud

        # Get and print distance value in mm at the center of the image
        # We measure the distance camera - object using Euclidean distance
        x = image.get_width() / 2
        y = image.get_height() / 2
        err, point_cloud_value = point_cloud.get_value(x, y)

        if err == sl.ERROR_CODE.SUCCESS:
            distance = math.sqrt(point_cloud_value[0]**2 + point_cloud_value[1]**2 + point_cloud_value[2]**2)
            print(f"Distance to Camera at {{{x};{y}}}: {distance} mm")
        i += 1

# Close the camera
zed.close()