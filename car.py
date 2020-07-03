import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import os
import shutil

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

def get_bboxes(image_RGB):
    original_image = image_RGB
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    # print(bboxes)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    return bboxes

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

scene = str(input('请输入摄像头序列号（01-48）：'))
path = 'scene/' + scene
path_mask = 'mask/' + scene

image_path_list, image_name_list = get_files(path)
mask_path_list, mask_name_list = get_files(path_mask)
obj_dic = {0:['人',0], 1:['非机动车',1,3], 2:['汽车',2,5,7]}
mask_dic = {0:'可停车区域', 1:'不可停车区域'}
judgement_matrix = [[0,1,1],      #可停车区域
                    [1,1,1]]        #不可停车区域
                #人，非机动车，汽车
print(image_path_list)
alarm = np.zeros(len(image_name_list))
alarm_range = 6     #可调
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
num_classes     = 80            #person=0，bike=1, car=2, motobike=3, truck=7
input_size      = 640
graph           = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:
    for img in range(len(image_path_list)):
        print('正在处理摄像头%s第%d张图...'%(scene, (img+1)))
        image = cv2.imread(image_path_list[img])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = get_bboxes(image)

        image = utils.draw_bbox(image, bboxes)
        image = Image.fromarray(image)
        image.show()

        estimate = get_obj_num(bboxes, mask_path_list, obj_dic)
        mask_num, obj_num = estimate.shape
        for i, obj in obj_dic.items():
            for j, mask in mask_dic.items():
                # print(judgement_matrix)
                # print(estimate)
                if judgement_matrix[j][i] ==1 and estimate[j, i] !=0:
                    print('在%s上有%d个%s' % (mask_dic[j], estimate[j, i], obj_dic[i][0]))
                    alarm[img] = 1

    print(alarm)
    if sum(alarm[-alarm_range:]) >= (alarm_range - 2):
        print('\n\n\n','*'*40,'\n\n警报警报警报！\n\n','*'*40)

