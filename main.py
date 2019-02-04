import sys
import time
import numpy as np
import utility as util
from operator import itemgetter

class instanceRecognation:
    def __init__(self):
        self.su = util.siftUtility()
        self.fu = util.fileUtility()
        self.mu = util.modelUtility()
        self.img_db = {}
        self.train_db = {}
        self.test_db = {}
        self.fm_results = {}
        self.cmap = {}
    
    def loadImageDatabase(self, img_db_file_path):
        self.img_db = self.fu.loadJSON(img_db_file_path)
        if 'train' in self.img_db:
            self.train_db = self.img_db['train']
        if 'test' in self.img_db:
            self.test_db = self.img_db['test']

    def extractSiftFeatures(self, img):
        corner_points = None
        if 'corners' in img:
            corner_points = img['corners']
        kp, des = self.su.features(img['path'], corner_points)
        return kp, des
    
    def processTrainImages(self):
        print("computing sift features for training images...")
        start_time = time.time()
        for img in self.train_db:
            kp, des = self.extractSiftFeatures(img)
            self.fu.dumpKeypoints(kp, img['class']+'_'+img['name'])
            self.fu.dumpDescriptors(des, img['class']+'_'+img['name'])
        print(str(len(self.train_db)), "images processed in %.3f seconds" % (time.time() - start_time))
    
    def processTestImages(self):
        print("computing sift features for testing images...")
        start_time = time.time()
        for img in self.test_db:
            kp, des = self.extractSiftFeatures(img)
            self.fu.dumpKeypoints(kp, img['name'])
            self.fu.dumpDescriptors(des, img['name'])
        print(str(len(self.train_db)), "images processed in %.3f seconds" % (time.time() - start_time))
    
    def featureMatchTest(self, reverse=False):
        print("comparing sift features between test and train images...")
        start_time = time.time()
        self.fm_results = {}
        for test_img in self.test_db:
            self.fm_results[test_img['name']] = []
            test_des = self.fu.loadDescriptors(test_img['name'])
            for train_img in self.train_db:
                train_des = self.fu.loadDescriptors(train_img['class']+'_'+train_img['name'])
                metric = self.su.BFDistanceFeatureMatch(train_des, test_des)
                self.fm_results[test_img['name']].append({'metric':metric, 'name':train_img['class']+'_'+train_img['name']})
            self.fm_results[test_img['name']] = sorted(self.fm_results[test_img['name']], key=itemgetter('metric'), reverse=reverse)
            self.fu.dumpFeatureMatchTestResults(self.fm_results[test_img['name']], test_img['name'])
        print(str(len(self.test_db)), "test images compared in %.3f seconds" % (time.time() - start_time))
    
    def trainClassificationModel(self):
        print("train for classification...")
        des = []
        labels = []
        cno = 0
        self.cmap = {}
        for img in self.train_db:
            if img['class'] not in self.cmap:
                self.cmap[img['class']] = cno
                self.cmap[cno] = img['class']
                cno+=1
            des.append(self.fu.loadDescriptors(img['class']+'_'+img['name']))
            labels.append(self.cmap[img['class']])
        print(len(des), "images")
        print("preprocessing completed")
        self.fu.dumpModelJSON(self.cmap, 'class_map')
        self.mu.stackDescriptors(des)
        self.mu.cluster()
        self.mu.generateEmbedding(des)
        self.mu.trainClassifier(labels)
    
    def testClassificationModel(self):
        for img in self.test_db:
            des = []
            des.append(self.fu.loadDescriptors(img['name']))
            self.mu.stackDescriptors(des)
            self.mu.generateEmbedding(des)
            prob = self.mu.testClassifier()

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("usage:\npython main.py path/to/image_description_file mode")
        sys.exit(0)
    
    ir = instanceRecognation()
    ir.loadImageDatabase(sys.argv[1])
    ir.processTrainImages()
    #ir.processTestImages()
    #ir.featureMatchTest()
    #ir.trainClassificationModel()
    ir.testClassificationModel()