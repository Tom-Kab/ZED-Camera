import pyzed.sl as sl
import sys

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 or HD1200 video mode (default fps: 60)
init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP # Use a right-handed Y-up coordinate system
init_params.coordinate_units = sl.UNIT.METER # Set units in meters

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)

# Enable positional tracking with default parameters
tracking_parameters = sl.PositionalTrackingParameters()
err = zed.enable_positional_tracking(tracking_parameters)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)

# Define runtime parameters
runtime_parameters = sl.RuntimeParameters()

i = 0
zed_pose = sl.Pose()
# Track the camera position during 1000 frames
while i < 1000:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Get the pose of the left eye of the camera with reference to the world frame
        zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)

        # Display the translation and timestamp
        py_translation = sl.Translation()
        tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
        ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
        tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
        timestamp = zed_pose.timestamp.get_milliseconds()

        # Display the orientation quaternion
        py_orientation = sl.Orientation()
        ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
        oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
        oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
        ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)

        # Print the updated values in the same line
        sys.stdout.write("\rTranslation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3} | Orientation: Ox: {4}, Oy: {5}, Oz {6}, Ow: {7}".format(tx, ty, tz, timestamp, ox, oy, oz, ow))
        sys.stdout.flush()
        
        i += 1

# Disable positional tracking and close the camera
zed.disable_positional_tracking()
zed.close()