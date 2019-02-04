import os
import json
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

class fileUtility:
    def __init__(self):
        self.db = {}

        self.data_path='data/'
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        
        self.sift_path='features/'
        if not os.path.exists(self.sift_path):
            os.makedirs(self.sift_path)
        
        self.models_path='models/'
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        self.results_path='results/'
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
    
    def loadJSON(self, img_db_file_path):
        try:
            with open(img_db_file_path,'r') as f:
                self.db = json.load(f)
        except Exception as e:
            print(e)
        return self.db
    
    def dumpDescriptors(self, des, filename):
        np.savetxt(self.sift_path+filename+'.csv', des, delimiter=',')
    
    def loadDescriptors(self, filename):
        data = np.loadtxt(self.sift_path+filename+'.csv', delimiter=',')
        return data

    def dumpKeypoints(self, kp, filename):
        index = []
        for point in kp:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            index.append(temp)
        try:
            with open(self.sift_path+filename+'.pickle', "wb") as f:
                pickle.dump(index, f)
        except Exception as e:
            print(e)
    
    def loadKeypoints(self, filename):
        kp=[]
        try:
            with open(self.sift_path+filename+'.pickle', "rb") as f:
                index = pickle.load(f)
        except Exception as e:
            print(e)
            return kp
        
        for point in index:
            temp = cv.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
            kp.append(temp)
        return kp
    
    def dumpFeatureMatchTestResults(self, score, name):
        with open(self.results_path+name+'.txt', "w") as f:
            for i in range(len(score)):
                f.write(matchness[i]['ic'] +'_'+ matchness[i]['image']+'.jpg\n')

class siftUtility:
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()
        self.bf = cv.BFMatcher()
    
    def generateMask(self, img_shape, corner_point):
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv.rectangle(mask, (corner_point[0][0],corner_point[0][1]), (corner_point[1][0],corner_point[1][1]), (255,255,255), thickness=-1)
        return mask

    def features(self, img_path, corner_points=None):
        image = cv.imread(img_path)
        if corner_points!=None:
            mask = self.generateMask(image.shape[:2], corner_points)
        else:
            mask = None
        
        kp, des = self.sift.detectAndCompute(image, mask)
        return kp, des
    
    def BFDistanceFeatureMatch(self, des1, des2, n=10):
        matches = self.bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        dist = 100000.0
        if len(matches) >= n:
            dist = 0
            for i in range(n):
                dist += matches[i].distance
        return dist
    
    def BFRatioTestFeatureMatch(self, des1, des2, t=0.75):
        matches = self.bf.knnMatch(des1,des2, k=2)
        good = 0
        for m,n in matches:
            if m.distance < t*n.distance:
                good+=1
        return good