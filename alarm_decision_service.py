import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import yaml
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def get_files(path):
    dirs = os.listdir(path)
    mask_dir_dict = {}
    for each in dirs:
        dir_path = os.path.join(path, each)
        if not os.path.isdir(dir_path):
            continue

        mask_dir_dict[each] = [os.path.join(dir_path, '0-mask-parking.jpg'), os.path.join(dir_path, '1-mask-unparking.jpg')]

    return mask_dir_dict


MAX_STATUS_LEN = 5
TOTAL_CAMERA_NUM = 48

app = FastAPI()

status_queues = [[] for i in range(TOTAL_CAMERA_NUM)]

with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

mask_dir_dict = get_files('mask/')
# print(mask_dir_dict)


class Item(BaseModel):
    camera_no: str
    objs: list

def params_parse(item: Item):
    camera_no = item.camera_no
    objs = item.objs
    obj_num = len(objs)
    # print(obj_num)
    bboxes = np.zeros((obj_num, 6), dtype=np.float)
    for i in range(obj_num):
        bboxes[i][0] = objs[i]['x1']
        bboxes[i][1] = objs[i]['y1']
        bboxes[i][2] = objs[i]['x2']
        bboxes[i][3] = objs[i]['y2']
        bboxes[i][4] = objs[i]['confidence']
        bboxes[i][5] = objs[i]['class']
    return camera_no, bboxes


def get_obj_num(bboxes, mask_path_list, obj_dic):
    ratio = 0.2
    estimate = np.zeros((len(mask_path_list), len(obj_dic)))       #(场地，物体）
    # [person, bike, car] #person=0，bike=1, car=2, motobike=3, truck=7
    for j, path in enumerate(mask_path_list):
        mask = cv2.imread(mask_path_list[j], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = (mask / 255).astype(int)
        for k, array in enumerate(bboxes):
            for i , obj in obj_dic.items():
                if array[5] in obj:
                    if np.sum(mask[int(0.5 * array[1] + 0.5 * array[3]):int(array[3]), int(array[0]):int(array[2])]) > 0.5 * ratio * (array[2]-array[0])* (array[3]-array[1]):
                        estimate[j, i] += 1

    return estimate


def decision_process(bboxes, mask_path_list, status_queue):

    obj_dict = cfg['obj_dict']
    mask_dict = cfg['mask_dict']
    estimate = get_obj_num(bboxes, mask_path_list, obj_dict)
    if estimate is None:
        return {'alarm':'mask_image_error!'}
    judgement_matrix = [[0,1,1], #可停车区域
                        [1,1,1]] #不可停车区域

    alarm = 0
    for i, obj in obj_dict.items():
        for j, mask in mask_dict.items():
            if judgement_matrix[j][i] == 1 and estimate[j, i] !=0:
                # print('在%s上有%d个%s' % (mask_dict[j], estimate[j, i], obj_dict[i][0]))
                alarm = 1

    status_queue.append(alarm)
    if len(status_queue) > MAX_STATUS_LEN:
        status_queue.pop(0)

    if sum(status_queue) > (cfg['alarm_range'] - 2):
        res = {'alarm': 1}
    else:
        res = {'alarm': 0}

    return res

@app.post("/alarm_decision_service")
async def alarm_decision_service(item: Item):

    try:
        camera_no, bboxes = params_parse(item)
    except:
        logging.error('params_parse error!')
        return -1
    
    # print(camera_no, bboxes)

    mask_path_list = mask_dir_dict[camera_no]
    status_queue = status_queues[int(camera_no)]
    
    ret = decision_process(bboxes, mask_path_list, status_queue)

    return ret



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="info")