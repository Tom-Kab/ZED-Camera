import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.AUTO  # Use HD720 or HD1200 video mode (default fps: 60)
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

# Enable spatial mapping with default parameters
mapping_parameters = sl.SpatialMappingParameters()
err = zed.enable_spatial_mapping(mapping_parameters)
if (err != sl.ERROR_CODE.SUCCESS):
    exit(-1)

# Grab data during 500 frames
i = 0
mesh = sl.Mesh() # Create a mesh object
while (i < 1000) :
    if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
        # In background, spatial mapping will use new images, depth and pose to create and update the mesh. No specific functions are required here
        mapping_state = zed.get_spatial_mapping_state()

        # Print spatial mapping state
        print("\rImages captured: {0} / 1000 || Mapping state: {1}".format(i, mapping_state), end="")
        i = i+1
print('\n')

zed.extract_whole_spatial_map(mesh) # Extract the whole mesh
mesh.filter(sl.MESH_FILTER.LOW) # Filter the mesh (remove unnecessary vertices and faces)
mesh.save("mesh.obj") # Save the mesh in an obj file

# Disable tracking and mapping and close the camera
zed.disable_spatial_mapping()
zed.disable_positional_tracking()
zed.close()