import pyzed.sl as sl

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.AUTO 
init_params.camera_fps = 30 

# Open the camera
err = zed.open(init_params)
if (err != sl.ERROR_CODE.SUCCESS) :
    exit(-1)
    
if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    while (i < 50) :
        # Grab an image
        if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp at the time the image was captured
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                timestamp.get_milliseconds()))
            i = i+1
            
# Close the camera
zed.close()