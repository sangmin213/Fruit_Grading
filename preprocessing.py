# 한 번에 실행하는 코드
from PIL import Image
import cv2
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

import detectron2
from detectron2.utils.logger import setup_logger

import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# import PointRend project
from detectron2.projects import point_rend

import argparse
import os

class preprocessing():
    def __init__(self):
        # create a Gaussing filter
        self.gaussian=GaussianBlur(kernel_size=(3,3),sigma=(0.01,0.01))
        # create a CLAHE object (Arguments are optional).
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.coco_metadata=MetadataCatalog.get("coco_2017_val")
        self.cfg=get_cfg()        
        point_rend.add_pointrend_config(self.cfg)
        # Load a config from file
        self.cfg.merge_from_file("./projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        self.cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
        self.predictor= DefaultPredictor(self.cfg)

    def Pointrend(self,img):
        outputs = self.predictor(img)
        # v = Visualizer(img[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:,:,::-1]  # -> draw pointrend segment contour
        mask = outputs["instances"].pred_masks  # -> draw instance mask 

        col = len(mask[0][0])
        row = len(mask[0])
        # backgroud -> black (color 값을 0으로 세팅) === instance에 해당하지 않는 영역을 모두 black으로 바꿔버림
        for i in range(row):
            for j in range(col):
                if(outputs["instances"].pred_masks[0][i][j] == False):
                    img[i,j] = 0

        return torch.Tensor(img)

    def Gaussian(self,img):
        blur_img = self.gaussian(img)
        return blur_img

    def Clahe(self,blur_img): 
        numpy_img=np.array(blur_img,dtype='uint8')  

        # r, g, b 채널별로 각각 적용
        r_img = numpy_img[:,:,0]
        g_img = numpy_img[:,:,1]
        b_img = numpy_img[:,:,2]

        r_clahe = self.clahe.apply(r_img)
        g_clahe = self.clahe.apply(g_img)
        b_clahe = self.clahe.apply(b_img)

        compressed_img=(np.dstack((r_clahe,g_clahe,b_clahe))).astype(np.uint8)

        return compressed_img

    def Meanshift(self,compressed_img):
        origin_shape = compressed_img.shape

        # Flatten image
        flatten_img = np.reshape(compressed_img,[-1,3])
        bandwidth = estimate_bandwidth(flatten_img,quantile=0.3,n_samples=1000)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(flatten_img)

        labels=ms.labels_
        meanshift_img=np.reshape(labels,origin_shape[:2])
        return meanshift_img

    def Kmeans(self,compressed_img):
        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = compressed_img.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)
        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # number of clusters (K)
        k = 5
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert back to 8 bit values
        centers = np.uint8(centers)
        # flatten the labels array
        labels = labels.flatten()
        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]
        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(compressed_img.shape)

        ''' masked img??? '''
        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(compressed_img)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        # color (i.e cluster) to disable
        cluster = 7
        masked_image[labels == cluster] = [0, 0, 0]
        # convert back to original shape
        masked_image = masked_image.reshape(compressed_img.shape)

        return segmented_image
        

    def train(self,img):
        # return self.meanshift(self.Clahe(self.Gaussian(self.Pointrend(img))))
        return self.Kmeans(self.Clahe(self.Gaussian(self.Pointrend(img))))


def argparser():
    parser = argparse.ArgumentParser(description='Parser for preprocessing fruit image to grading classification')

    parser.add_argument('--load_dir', '-L', required=True, help="directory path to load raw image")
    parser.add_argument('--save_dir', '-S', required=True, help="directory path to save processed image")

    args = parser.parse_args()

    return args        


def main():
    args = argparser()
    base_path=args.load_dir
    save_path=args.save_dir
    setup_logger() # for Using detectron2
    preprocesser=preprocessing()
    # detectron 적용 실패한 이미지 목록 저장
    f = open('../detectron_fail.txt',"w")

    for image_name in os.listdir(base_path):
        img=cv2.imread(base_path+'/'+image_name)
        try:
            img=preprocesser.train(img=img)
        except:
            f.write(image_name)
            continue
        cv2.imwrite(save_path+'/'+image_name,img)
    
    f.close()

if __name__ == '__main__' :
    main()