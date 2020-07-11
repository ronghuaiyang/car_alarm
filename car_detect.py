import cv2
import numpy as np
import core.utils as utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from PIL import Image
import os
import shutil
import yaml
import requests


class CarDetect():

    def __init__(self, cfg):

        self.cfg = cfg
        self.graph = tf.Graph()
        self.model = utils.read_pb_return_tensors(self.graph, self.cfg['pb_file'], self.cfg['return_elements'])
        self.sess = tf.Session(graph=self.graph)

    def infer(self, input_data):
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run([self.model[1], self.model[2], self.model[3]],
                                                            feed_dict={self.model[0]: input_data})
        return pred_sbbox, pred_mbbox, pred_lbbox


def get_files(path):
    path_list = []
    name_list = []
    #判断路径是否存在
    if (os.path.exists(path)):
    #获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for file in files:
            #得到该文件下所有目录的路径
            m = os.path.join(path,file)
            #判断该路径下是否是文件夹
            if (os.path.isfile(m)):
                p = os.path.abspath(m)
                n = os.path.split(m)
                path_list.append(p)
                name_list.append(n[1])
                path_list.sort()
                name_list.sort()
    return path_list,name_list



def get_latest_image(dir_path):
    file_lists = os.listdir(dir_path)
    file_lists.sort(key=lambda fn: os.path.getmtime(os.path.join(dir_path, fn))
                    if not os.path.isdir(os.path.join(dir_path, fn)) else 0)
    latest_file = os.path.join(dir, file_lists[-1])
    return latest_file

def get_obj_num(bboxes, mask_path_list, obj_dic):
    ratio = 0.2
    estimate = np.zeros((len(mask_path_list), len(obj_dic)))       #(场地，物体）
    # [person, bike, car] #person=0，bike=1, car=2, motobike=3, truck=7
    for j, path in enumerate(mask_path_list):
        mask = cv2.imread(mask_path_list[j], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255).astype(int)
        for k, array in enumerate(bboxes):
            for i , obj in obj_dic.items():
                # if array[5] == 1:
                #     print('*'*20, array)
                if array[5] in obj:
                    if np.sum(mask[int(0.5 * array[1] + 0.5 * array[3]):int(array[3]), int(array[0]):int(array[2])]) > 0.5 * ratio * (array[2]-array[0])* (array[3]-array[1]):
                        estimate[j, i] += 1

    return estimate


# def car_alarm(car_detect, cfg, mask_path_list, alarm_list):

#     img_file = get_latest_image(cfg['scene/' + cfg['camera_no']])
#     image = cv2.imread(img_file)
#     if image is None:
#         return

#     # input_data transform
#     input_size = cfg['input_size']
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     original_image = image
#     original_image_size = original_image.shape[:2]
#     image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
#     image_data = image_data[np.newaxis, ...]

#     # model infer
#     pred_sbbox, pred_mbbox, pred_lbbox = car_detect.infer(image_data)

#     # post process
#     num_classes = cfg['num_classes']
#     pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
#                                 np.reshape(pred_mbbox, (-1, 5 + num_classes)),
#                                 np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
#     bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
#     bboxes = utils.nms(bboxes, 0.45, method='nms')

#     obj_dict = cfg['obj_dict']
#     mask_dict = cfg['mask_dict']
#     estimate = get_obj_num(bboxes, mask_path_list, obj_dict)
#     judgement_matrix = [[0,1,1], #可停车区域
#                         [1,1,1]] #不可停车区域

#     mask_num, obj_num = estimate.shape
#     alarm = 0
#     for i, obj in obj_dict.items():
#         for j, mask in mask_dict.items():
#             if judgement_matrix[j][i] ==1 and estimate[j, i] !=0:
#                 # print('在%s上有%d个%s' % (mask_dict[j], estimate[j, i], obj_dict[i][0]))
#                 alarm = 1
#             else:
#                 alarm = 0
    
#     alarm_list.append(alarm)
#     alarm_list.pop(0)

#     if sum(alarm_list) >= cfg['alarm_range'] - 2:
#         # call http service
#         url = cfg['url']
#         data = {"alram": 1}
#         ret = requests.post(url, data=data)


def car_alarm(image, car_detect, cfg, mask_path_list, alarm_list):

    # img_file = get_latest_image(cfg['scene/' + cfg['camera_no']])
    # image = cv2.imread(img_file)
    # if image is None:
    #     return

    # input_data transform
    # 
    input_size = cfg['input_size']
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    # model infer
    pred_sbbox, pred_mbbox, pred_lbbox = car_detect.infer(image_data)

    # post process get final bboxes
    num_classes = cfg['num_classes']
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    # set judgement_matrix and estimate
    obj_dict = cfg['obj_dict']
    mask_dict = cfg['mask_dict']
    estimate = get_obj_num(bboxes, mask_path_list, obj_dict)
    judgement_matrix = [[0,1,1], #可停车区域
                        [1,1,1]] #不可停车区域

    alarm = 0
    for i, obj in obj_dict.items():
        for j, mask in mask_dict.items():
            if judgement_matrix[j][i] == 1 and estimate[j, i] !=0:
                # print('在%s上有%d个%s' % (mask_dict[j], estimate[j, i], obj_dict[i][0]))
                alarm = 1

    res = {'alarm': alarm}

    objs = []
    for bbox in bboxes:
        objs.append({'x1':bbox[0], 'y1':bbox[1], 'x2':bbox[2], 'y2':bbox[3], 'confidence':bbox[4], 'class':bbox[5]})
    res['objs'] = objs 

    return res



    






    


    