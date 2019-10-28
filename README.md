# TRAFFIC ANALYSIS WITH TRAFFIC CAMERA

1. Vehicles count for each lane
2. Pedestrians in either direction
3. Video time of entry of road user

The given dataset has 4953 frames at ~10fps with (width,height) = (1280,720)

### TODO
- [x] Extract images from video
- [x] Detect all types of vehicles in the video
- [x] Counting vehicles
- [x] Pedestrian detection
- [x] Track pedestrians over crossing and count them
- [x] Stabilize the count of vehicles
- [ ] Add time information
- [ ] Find direction of pedestrian crossing 

### FAILURE CASES
1. Pedestrian on the sidewalk but far
2. Bicycles and motorcycles - people get detected multiple times

### USAGE
All the code has been developed and tested on Linux Ubuntu 16.04 with 410.104  Nvidia driver and Cuda version 10.0 on GeForce GTX 1050. Assuming the videos are taken from the same camera facing the same road, similar to the given video. 

**Dependencies**
1. Opencv (version>3)
2. Keras
3. Tensorflow gpu (version<2)
4. Numpy
5. Tesseract-ocr
6. Numba
7. Skimage
8. PyTorch
   

<!-- **Method 1** 

Perform object detection for both vehicles and pedestrians. Maintain count when entering, present and moving out of the frame. Assumption is that the traffic flow is in horizontal direction and a zebra crossing is along vertical.
```
git clone git@github.com:meenakshiravisankar/keras-yolo3.git
cd keras-yolo3
wget https://pjreddie.com/media/files/yolov3.weights
python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```
To run
```
python3 yolo_video.py --input <path-to-video>
```

The above generates a video with detection and counts traffic participants. It also generates log file with the time at which they are counted. The computation for object detection is ~0.23s per frame. -->

**Method**

Track the traffic participants continuously through the frames for a stable count. The only assumption here is that we know the region of zebra crossing (this can be acheived with a detection pipeline) to detect pedestrians crossing and ignore the rest of the people walking. 

Run the following commands
```
git clone git@github.com:meenakshiravisankar/pytorch_objectdetecttrack.git
cd pytorch_objectdetecttrack
bash setup.sh
cd config
bash download_weights.sh
```
To run the script
```
python3 traffic_tracker.py --input <path-to-video> --output(optional) <path-to-video> --activity(optional) <directory-to-save-log>
```
### CREDITS
1. Yolov3 - [link](https://github.com/pjreddie/darknet)
2. Keras-yolo3 - [link](https://github.com/meenakshiravisankar/keras-yolo3)
3. Pytorch tracking with yolo3 - [link](https://github.com/cfotache/pytorch_objectdetecttrack)
   
<!-- Original Readme.md -->
<!-- # PyTorch Object Detection and Tracking
Object detection in images, and tracking across video frames

Full story at:
https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98

References:
1. YOLOv3: https://pjreddie.com/darknet/yolo/
2. Erik Lindernoren's YOLO implementation: https://github.com/eriklindernoren/PyTorch-YOLOv3
3. YOLO paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf
4. SORT paper: https://arxiv.org/pdf/1602.00763.pdf
5. Alex Bewley's SORT implementation: https://github.com/abewley/sort
6. Installing Python 3.6 and Torch 1.0: https://medium.com/@chrisfotache/getting-started-with-fastai-v1-the-easy-way-using-python-3-6-apt-and-pip-772386952d03 -->
