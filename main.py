import sys
import time
import utility as util
from operator import itemgetter

class instanceRecognation:
    def __init__(self):
        self.su = util.siftUtility()
        self.fu = util.fileUtility()
        self.img_db = {}
        self.train_db = {}
        self.test_db = {}
        self.fm_results = {}
    
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
            self.fu.dumpKeypoints(kp, img['name'])
            self.fu.dumpDescriptors(des, img['name'])
        print(str(len(self.train_db)), "images processed in %.3f seconds", (time.time() - start_time()))
    
    def processTestImages(self):
        print("computing sift features for testing images...")
        start_time = time.time()
        for img in self.test_db:
            kp, des = self.extractSiftFeatures(img)
            self.fu.dumpKeypoints(kp, img['name'])
            self.fu.dumpDescriptors(des, img['name'])
        print(str(len(self.train_db)), "images processed in %.3f seconds" % (time.time() - start_time()))
    
    def featureMatchTest(self, reverse=False):
        self.fm_results = {}
        for test_img in self.test_db:
            self.fm_results[test_img['name']] = []
            test_des = self.fu.loadDescriptors(test_img['name'])
            for train_img in self.train_db:
                train_des = self.fu.loadDescriptors(train_img['name'])
                metric = self.su.BFDistanceFeatureMatch(train_des, test_des)
                self.fm_results[test_img['name']].append({'metric':metric, 'name':train_img['name']})
            self.fm_results[test_img['name']] = sorted(self.fm_results[test_img['name']], key=itemgetter('metric'), reverse=reverse)
            self.fu.dumpFeatureMatchResults(self.fm_results[test_img['name']], test_img['name'])

if __name__ == '__main__':
    if len(sys.argv)!=2:
        print("usage:\npython main.py path/to/image_description_file mode")
        sys.exit(0)
    
    ir = instanceRecognation()
    ir.loadImageDatabase(sys.argv[1])
    ir.processTrainImages()
    ir.featureMatchTest()

    #if sys.argv[2]==0: