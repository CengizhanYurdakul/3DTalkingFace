import os
import cv2
import torch
import pickle
import numpy as np
import pyfakewebcam
from time import time
from tqdm import tqdm
from PIL import Image

from models import create_model
from util.load_mats import load_lm3d
from util.preprocess import align_img
from Detection.Detector import Detector
from options.test_options import TestOptions

from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

import tkinter as tk

print("CUDA:", torch.cuda.is_available())

class Configs:
    def __init__(self, root, targetName="seed0000.png", cameraAngle=180, cameraAngle2=0, cameraAngle3=2):
        
        self.target = tk.StringVar(root)
        self.target.set(targetName)
        self.targetButton = tk.OptionMenu(root, self.target, *os.listdir("Targets"))
        self.targetButton.pack()
        
        
        self.angle = tk.Scale(root, from_=0, to=359, orient=tk.HORIZONTAL)
        self.angle.set(cameraAngle)
        self.angle.pack()
        
        self.angle2 = tk.Scale(root, from_=0, to=359)
        self.angle2.set(cameraAngle2)
        self.angle2.pack()
        
        self.angle3 = tk.Scale(root, from_=1, to=10, resolution=0.5)
        self.angle3.set(cameraAngle3)
        self.angle3.pack()

class Processor:
    def __init__(self):
        self.opt = TestOptions().parse()
        self.device = "cuda:0"
        
        self.size = 512
        
        #TODO implement new face detection instead of `Detector` that gives 5 landmarks as an output
        self.faceDetector = Detector()
        
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.device = self.device
        self.model.eval()
        
        self.lm3d_std = load_lm3d("BFM")
        
        self.cameraAngle = 180
        self.cameraAngle2 = 0
        self.cameraAngle3 = 2
        
        self.initCamera(self.cameraAngle3, self.cameraAngle2, self.cameraAngle)

        self.raster_settings = RasterizationSettings(
            image_size=self.size, 
            blur_radius=0.0
        )

        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 0.0]])
        
        self.initRenderer(self.cameras)

        self.isFirst = True
        
        self.root = tk.Tk()  
        self.root.title("TalkingFace") 
        
        frame = tk.Frame(self.root)
        frame.pack()
        
        self.targetName = "seed0000.png"
        
        
        self.configs = Configs(self.root, self.targetName, self.cameraAngle)
        
    def initRenderer(self, cameras):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights
            )
        )
        
    def initCamera(self, dist, elev, azim):
        azim = int(azim)
        elev = int(elev)
        dist = int(dist)
        R, T = look_at_view_transform(dist, elev, azim) 
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        
    def readTransformImage(self, image):
        #TODO landmark variable is numpy array with shape (5, 2) that inclued eyes, nose and mouth points
        landmark = self.faceDetector.detect(image)["Landmark"][0]
        image = Image.fromarray(image)
        W, H = image.size
        landmark = landmark.reshape([-1, 2])
        landmark[:, -1] = H - 1 - landmark[:, -1]
        _, image, landmark, _ = align_img(image, landmark, self.lm3d_std)
        
        imageTensor = torch.tensor(np.array(image)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        landmarkTensor = torch.tensor(landmark).unsqueeze(0)
        return imageTensor, landmarkTensor
    
    def render(self, vertex, faces, color):
        
        
        textures = TexturesVertex(verts_features=torch.clip(color, 0, 1).to(self.device))
        mesh = Meshes(
            verts=[(vertex[0]).to(self.device)],   
            faces=[faces.to(self.device)],
            textures=textures
        )
        images = self.renderer(mesh)
        images = images[0, ..., :3].cpu().numpy()
        return images
    
    def renderFace(self, inputImage, targetImage, changeTarget):
        
        inputImageTensor, inputLandmarkTensor = self.readTransformImage(inputImage)
        if self.isFirst or changeTarget:
            self.targetImageTensor, self.targetLandmarkTensor = self.readTransformImage(targetImage)
            self.model.set_input({"imgs": self.targetImageTensor, "lms": self.targetLandmarkTensor})
            self.targetCoeffs = self.model.testCoeffs()
            self.isFirst = False
        
        self.model.set_input({"imgs": inputImageTensor, "lms": inputLandmarkTensor})
        inputCoeffs = self.model.testCoeffs()
        
        self.targetCoeffs[:, 80: 144] = inputCoeffs[:, 80: 144]
        
        vertex, texture, color, landmark, faces = self.model.renderCoeff(self.targetCoeffs)
        
        vertex[:, :, 2] = vertex[:, :, 2] - (vertex[:, :, 2].max()/1.1)
        
        renderedFace = self.render(vertex, faces, color)
        
        return renderedFace
    
    def inference(self):
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        
        targetImage = cv2.imread(os.path.join("Targets", self.targetName))
        
        stream = pyfakewebcam.FakeWebcam('/dev/video9', self.size, self.size)
        
        cap = cv2.VideoCapture(0)
        
        while(cap.isOpened()):
            self.root.update()
            ret, frame = cap.read()
            if ret == True:
                s = time()
                
                changeTarget = False
                if self.targetName != self.configs.target.get():
                    self.targetName = self.configs.target.get()
                    targetImage = cv2.imread(os.path.join("Targets", self.targetName))
                    changeTarget = True
                    
                if self.cameraAngle != self.configs.angle.get() or self.cameraAngle2 != self.configs.angle2.get() or self.cameraAngle3 != self.configs.angle3.get():
                    self.cameraAngle = self.configs.angle.get() # azim
                    self.cameraAngle2 = self.configs.angle2.get() # elev
                    self.cameraAngle3 = self.configs.angle3.get() # dist
                    self.initCamera(self.cameraAngle3, self.cameraAngle2, self.cameraAngle)
                    self.initRenderer(self.cameras)
                
                renderedFace = self.renderFace(frame, targetImage, changeTarget=changeTarget)
                
                stream.schedule_frame((cv2.cvtColor(renderedFace*255, cv2.COLOR_RGB2BGR)).astype(np.uint8))
                
                cv2.imshow('Frame', (cv2.cvtColor(renderedFace*255, cv2.COLOR_RGB2BGR)).astype(np.uint8))
                print("FPS: ", round(1 / (time()-s), 4))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    
if __name__ == '__main__':
    processor = Processor()
    processor.inference()