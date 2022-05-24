# 3D TalkingFace

![Test](assets/vis.gif)


**3DTalkingFace** for video calls.

**3DTalkingFace** is a project that can be used during video calls, reconstructing the face you specified in 3D and making it talk to the face in the image taken from the webcam.

# Installation
**3DTalkingFace** requires manually installation for some additional stuffs like [nvdiffrast](https://nvlabs.github.io/nvdiffrast/) and (insightface)[https://github.com/deepinsight/insightface]. Additional installation information can be found in the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) repository.


## Build Dockerfile
```
docker build -t talkingface .
```

## Run Image
```
docker run -it --runtime=nvidia --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --network=host --device="/dev/video0:/dev/video0" --device=/dev/video9:/dev/video9 -v ${PWD}:/app talkingface
```

## Important!
### **Since I am working on face detection in another project, I cannot share it at the moment, but you can easily implement your own face detection network and make the system usable with only 5 landmark outputs. In the `main.py` file, detection initialization and inference parts are specified as #TODO on lines 62 and 121.**