import os
import json
import joblib
import pickle
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
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
    
    def loadJSON(self, path):
        try:
            with open(path,'r') as f:
                self.db = json.load(f)
        except Exception as e:
            print(e)
        return self.db
    
    def dumpJSON(self, data, path):
        try:
            with open(path,'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(e)

    def dumpDescriptors(self, des, filename):
        np.save(self.sift_path+filename+'.npy', des)
    
    def loadDescriptors(self, filename):
        data = np.load(self.sift_path+filename+'.npy')
        return data.astype(np.float32)

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
    
    def dumpModelMatrices(self, data, filename):
        np.savetxt(self.data_path+filename+'.npy', data)
    
    def loadModelMatrices(self, filename):
        return np.loadtxt(self.data_path+filename+'.npy')
    
    def dumpModelJSON(self, data, filename):
        path = self.data_path+filename+'.json'
        self.dumpJSON(data, path)
    
    def loadModelJSON(self, filename):
        path = self.data_path+filename+'.json'
        return self.loadJSON(path)

    def dumpFeatureMatchTestResults(self, score, name):
        with open(self.results_path+name+'.txt', "w") as f:
            for i in range(len(score)):
                f.write(score[i]['name']+'.jpg\n')

class siftUtility:
    def __init__(self):
        self.sift = cv.xfeatures2d.SIFT_create()
        self.bf = cv.BFMatcher()
    
    def showKeypointsImage(self, img, kp):
        img=cv.drawKeypoints(img,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img)
        plt.show()
    
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
        #self.showKeypointsImage(image, kp)
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

class modelUtility():
    def __init__(self, clusters=100):
        self.n_clusters = clusters
        self.fu = fileUtility()
        self.des_stack = None
        self.centers = None
        self.embedding = None
        self.labels = []
        self.kmeans = None
        self.clf = None
    
    def stackDescriptors(self, descriptors):
        self.des_stack = np.vstack(d for d in descriptors)
        self.fu.dumpModelMatrices(self.des_stack, "des_stack")
    
    def cluster(self, batch_size=2000):
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=batch_size, verbose=True)
        self.centers = self.kmeans.fit_predict(self.des_stack)
        self.fu.dumpModelMatrices(self.centers, "cluster_centers")
        joblib.dump(self.kmeans, self.fu.models_path+'kmeans.joblib')
    
    def obtainCenters(self):
        self.kmeans = joblib.load(self.fu.models_path+'kmeans.joblib')
        self.centers = self.kmeans.predict(self.des_stack)

    def generateEmbedding(self, des):
        self.embedding = np.zeros((len(des),self.n_clusters))
        pt = 0
        for i in range(len(des)):
            m = len(des[i])
            for j in range(m):
                self.embedding[i][self.centers[pt+j]]+=1
            pt+=m
    
    def trainClassifier(self, labels):
        self.labels = labels
        self.clf = SVC(probability=True, verbose=True)
        self.clf.fit(self.embedding, self.labels)
        joblib.dump(self.clf, self.fu.models_path+'svc.joblib')
    
    def testClassifier(self):
        self.clf = joblib.load(self.fu.models_path+'svc.joblib')
        return self.clf.predict_proba(self.embedding)