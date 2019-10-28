import pytesseract
# from driver import *
from utils.utils import *
from utils import *
from models import *

import os, sys, time, datetime, random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import cv2

def get_videotime(frame) :
    """
    Crops the time-in-video and returns the timestamp

    """
    # Crop the time portion
    frame = frame[0:25,0:220]
    # perform ocr
    text = "dummy"
    # text = pytesseract.image_to_string(frame)  
    return text

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80,
                        conf_thres, nms_thres)
    return detections[0]

# Config parameters
config_path='./config/yolov3.cfg'
weights_path='./config/yolov3.weights'
class_path='./config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor

# path to the video
videopath = '../files/videos/1569843500.mp4'

cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
from sort import *
vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()

# Initialising count of traffic participants
current_vehicles = 0
total_left_vehicles = 0
total_right_vehicles = 0

current_pedestrians = 0
total_pedestrians = 0

activity_log = np.array([])
save_log_time = 0
activity_log_path = "../files/results/"

vehicle_classes = [2,3,5,7]
pedestrian_classes = [0]

if not os.path.exists(activity_log_path) :
    os.makedirs(activity_log_path)

participants_id = []

while(True):
    ret, frame = vid.read()
    if not ret :
        break
    time_in_video = get_videotime(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        
        # get vehicles and pedestrians separately
        mask = np.isin(tracked_objects[:,-1], vehicle_classes)
        vehicle_labels = tracked_objects[mask]
        mask = np.isin(tracked_objects[:,-1], pedestrian_classes)
        pedestrian_labels = tracked_objects[mask]

        height = img.shape[0]
        width = img.shape[1]
        # for pedestrians
        req_height = int(height/2)-140

        current_vehicles = vehicle_labels.shape[0]
        current_pedestrians = pedestrian_labels.shape[0]
        
        for x1, y1, x2, y2, obj_id, cls_pred in vehicle_labels:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),
                         color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,
                         y1), color, -1)
            
            col = int((x1*2+box_w)/2)
            row = int((y1*2+box_h)/2)

            if obj_id not in participants_id :
                participants_id.append(obj_id)
                if row>height/2 :
                    total_left_vehicles+=1
                else :
                    total_right_vehicles+=1
                vehicle = np.array([x1,y1,x1+box_w,y1+box_h,time_in_video])   
                activity_log = np.append(activity_log, vehicle)
            
            cv2.putText(frame, cls, 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255,255,255), 3)
            
        for x1, y1, x2, y2, obj_id, cls_pred in pedestrian_labels:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            
            col = int((x1*2+box_w)/2)
            row = int((y1*2+box_h)/2)

            if row > req_height :  
                if obj_id not in participants_id :
                    participants_id.append(obj_id)
                    total_pedestrians+=1
                    pedestrian = np.array([x1,y1,x1+box_w,y1+box_h,time_in_video])   
                    activity_log = np.append(activity_log, pedestrian)
                
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h),
                         color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60,
                         y1), color, -1)
                cv2.putText(frame, cls, 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 3)

    # Figure for counts
    size = 256
    font_size = 0.6
    img = np.zeros((size,size,3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,'Vehicles',(5,30), font, font_size,(255,255,255),2,8)
    # cv2.putText(img,'Current Frame : '+str(current_vehicles),(0,60), font, font_size,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,'Left Lane : '+str(total_left_vehicles),(5,70), font, font_size,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,'Right Lane : '+str(total_right_vehicles),(5,110), font, font_size,(255,255,255),2,cv2.LINE_AA)
    img = cv2.line(img,(5,130),(100,130),(255,255,255),1)

    cv2.putText(img,'Pedestrians',(5,160), font, font_size,(255,255,255),2,8)
    # cv2.putText(img,'Current Frame : '+str(current_pedestrians),(5,180), font, font_size,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(img,'Total   : '+str(total_pedestrians),(5,200), font, font_size,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(img,'Total   : '+str(total_pedestrians),(5,240), font, font_size,(255,255,255),2,cv2.LINE_AA)
    
    frame = np.array(frame)
    frame[0:size,width-size:width] = img
        
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("TRAFFIC ANALYSIS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
