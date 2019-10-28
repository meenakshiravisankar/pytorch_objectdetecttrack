import cv2
import os
import pytesseract

def get_property(path_to_video) :
    """
    Computes properties given the path of the video
    
    Parameters
    ----------
    path_to_video - path to the mp4 file
   
    Returns
    -------
    fps - frames per second
    frames - total frames in video
    width - width of the frame
    height - height of the frame
    """
    video = cv2.VideoCapture(path_to_video)    
    fps = video.get(cv2.CAP_PROP_FPS)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    video.release()
    return int(fps), int(frames), int(width), int(height)

def display_video(path_to_video) : 
    """
    Displays video
    
    Parameters
    ----------
    path_to_video - path to the mp4 file
    
    Returns
    -------
    NA
    """
    video = cv2.VideoCapture(path_to_video)
    while(video.isOpened()) :
        _ , frame = video.read()        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        fps, _, _, _ = get_property(path_to_video)
        wait_ms = int(1000/fps)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q') :
            break
    video.release()
    cv2.destroyAllWindows()

def extract_images(path_to_video, path_to_images) :
    """
    Extract images from video
    
    Parameters
    ----------
    path_to_video - path to the mp4 file
    path_to_images - path to the directory to save images
    
    Returns
    -------
    NA
    """
    count = 0    
    if not os.path.exists(path_to_images):
        os.makedirs(path_to_images)
    video = cv2.VideoCapture(path_to_video)
    while(video.isOpened()) :
        ret, frame = video.read()
        if not ret :
            break       
        image_name = str(count)+".jpg"
        cv2.imwrite(os.path.join(path_to_images, image_name), frame)
        count+=1
    video.release()
    

def create_video(path_to_images, path_to_video, fps, seconds) :
    """
    Saves video of the length equal to seconds

    Parameters
    ----------
    path_to_images - directory that contains images of full video
    path_to_video - path to video file that has to be saved
    fps - frames per second
    seconds - number of seconds to save
    
    Returns
    -------
    NA
    """
    frames = seconds*fps
    image_count = 0
    image_path = os.path.join(path_to_images, str(image_count)+".jpg")
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(path_to_video, fourcc, fps, (width, height))

    while(frames):
        image_path = os.path.join(path_to_images, str(image_count)+".jpg")
        image = cv2.imread(image_path)

        video.write(image)
        frames-=1 
        image_count+=1

    # Release everything if job is finished
    video.release()



def get_videotime(frame) :
    """
    Crops the time-in-video and returns the timestamp

    Parameters
    ----------
    frame - image frame
    
    Returns
    -------
    text - timestamp present on the image 

    """
    # Crop the time portion
    frame = frame[0:25,0:220]
    # perform ocr
    text = pytesseract.image_to_string(frame)  
    return text
    
# Sample paths

path_to_video = "../files/videos/1569843500.mp4"
path_to_new_video = "../files/videos/raw60.mp4"
path_to_images = "../files/images"
path_to_image = "../files/images/0.jpg"

