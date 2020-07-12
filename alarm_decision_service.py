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
# from car_detect import CarDetect, car_alarm, get_files

MAX_STATUS_LEN = 5
TOTAL_CAMERA_NUM = 48

app = FastAPI()

status_queues = [[] for i in range(TOTAL_CAMERA_NUM)]


class Item(BaseModel):
    camera_no: str
    objs: list

def params_parse(item: Item):
    camera_no = item.camera_no
    objs = item.objs
    return int(camera_no), objs


def decision_process():
    # it's the decision function
    # choose the information you need from status_queues
    # process to get the result
    # TODO
    return 1

@app.post("/alarm_decision_service")
async def alarm_decision_service(item: Item):

    try:
        camera_no, objs = params_parse(item)
    except:
        logging.error('params_parse error!')
        return -1

    # add objs to list 
    # if the queue is full pop the oldest status
    status_queues[0].append(objs)
    if len(status_queues[camera_no]) > MAX_STATUS_LEN:
        status_queues[camera_no].pop(0)

    # scan the queue
    print(status_queues)
    for status_queue in status_queues:
        for objs in status_queue:
            print(objs)

    ret = decision_process()

    return ret



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5001, log_level="info")