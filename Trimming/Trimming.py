#outlines 
import torch as pt
import numpy as np
import cv2
import os

DEVICE = "cuda:0"

class FFModel(pt.nn.Module):
    def __init__(self):
        super(FFModel, self).__init__()
        self.activation = pt.nn.ReLU()
        self.sig = pt.nn.Sigmoid()
        self.flatten = pt.nn.Flatten()
        self.conv1 = pt.nn.Conv2d(1,128,(3,3),1)
        self.conv2 = pt.nn.Conv2d(128,32,(3,3),1)
        self.conv3 = pt.nn.Conv2d(32,8,(3,3),1)
        self.conv4 = pt.nn.Conv2d(8,4,(3,3),1)
        self.convArry = [self.conv1,self.conv2,self.conv3,self.conv4]
        self.maxPool = pt.nn.MaxPool2d((2,2),2)
        self.linear1 = pt.nn.Linear(1404,512)
        self.linear2 = pt.nn.Linear(512,128)
        self.linear3 = pt.nn.Linear(128,1)
        self.linearArry = [self.linear1,self.linear2,self.linear3]
        
    def forward(self, x):
        for conv in self.convArry:
            x = conv(x)
            x = self.activation(x)
            x = self.maxPool(x)
        x = self.flatten(x)
        for linear in self.linearArry:
            x = linear(x)
            if linear != self.linearArry[-1]:
                x = self.activation(x)
        x = self.sig(x)
        return x

#runs prediciton on light probability given an img
def run_prediction(img):
    #loading model from pt file
    test_model = pt.load("lightDetector.pt")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.pyrDown(img)
    X = np.array(img).reshape(1, 1, *img.shape)
    X = pt.tensor(X, dtype=pt.float32).to(DEVICE)/255

    return test_model(X).item()

#alternative method that checks pixel values to see if light is on
def light_is_on(frame, threshold):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rows, cols = grayFrame.shape
    rows = int(rows/2)
    cols = int(cols/2)
    k = 0
    total = 0
    for i in range(rows):
        for j in range(cols):
            k += grayFrame.item(i, j)
            total += 1

    total = k / total

    if (total < threshold):
        return True
    return False

#loads the videos from RawVideo Folder
def load_videos_from_folder(folder):
    videos = []
    for filename in os.listdir(folder):
        path = os.path.join(folder,filename)
        if path is not None:
            videos.append(path)
    return videos


# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames, image = vidcap.read()
#     if hasFrames:
#         return image
#     return hasFrames


from moviepy.editor import *
#makes a subclip that includes 20 seconds before light and 30 seconds after light
def makeClip(vid, frameNumber, vid_index, clipNumber, cap):

    # loading video dsa gfg intro video 
    clip = VideoFileClip(vid) 
        
    # getting only first 5 seconds
    # 30 fps

    startTime = round((frameNumber - 600) / fps)
    endTime = round((frameNumber + 900) / fps) 
    clip = clip.subclip(startTime, endTime) 
    
    # cutting out some part from the clip
    
    # showing  clip
    filename = 'video-' + str(vid_index) + '_clip-' + str(clipNumber) + '.avi'
    clip.write_videofile(os.path.join(out_path , filename), codec='rawvideo')



if __name__ == "__main__": 

    in_path = './RawVideo'
    videos = load_videos_from_folder(in_path)

    fps = 30.0
    out_path = 'C:/Users/Michael/RatPosture/Trimming/ProcessedVideo'

    for vid_index, vid in enumerate(videos):
        # Opens the Video file
        cap = cv2.VideoCapture(vid)
        
        frameNumber = 0
        clipNumber = 0
        time = 0
        while (cap.isOpened()):
            clipNumber += 1
            prediction = 0
            ret, frame = cap.read()
            if ret == False:
                break

            while (prediction < 0.6):
                frameNumber += 300
                time += 10000
                cap.set(cv2.CAP_PROP_POS_MSEC, time)
                ret, frame = cap.read()
                if ret == False:
                    break
                prediction = run_prediction(frame)

            if ret == False:
                break
            
            time -= 10000
            frameNumber -= 300
            cap.set(cv2.CAP_PROP_POS_MSEC, time)    

            prediction = 0
            while (prediction < 0.6):
                frameNumber += 30
                time += 1000
                cap.set(cv2.CAP_PROP_POS_MSEC, time)
                ret, frame = cap.read()
                if ret == False:
                    break
                
                prediction = run_prediction(frame)
                cv2.imshow('frame', frame)
                cv2.waitKey(100)
                
            makeClip(vid, frameNumber, vid_index, clipNumber, cap)

            time += 20000
            frameNumber += 600
            cap.set(cv2.CAP_PROP_POS_MSEC, time)
            
            

        cap.release()
        cv2.destroyAllWindows()


