import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
import core.utils as utils


return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "./yolov3_coco.pb"
class ObjDetectInfer():

    def __init__(self, cfg):

        self.cfg = cfg
        self.graph = tf.Graph()
        self.model = utils.read_pb_return_tensors(self.graph, pb_file, return_elements)
        self.sess = tf.Session(graph=self.graph)

    def infer(self, input_data):
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run([self.model[1], self.model[2], self.model[3]],
                                                            feed_dict={self.model[0]: input_data})
        return pred_sbbox, pred_mbbox, pred_lbbox


app = FastAPI()

with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

obj_detect_infer = ObjDetectInfer(cfg)


def obj_detect(image):

    input_size = cfg['input_size']
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    # model infer
    pred_sbbox, pred_mbbox, pred_lbbox = obj_detect_infer.infer(image_data)

    # post process get final bboxes
    num_classes = cfg['num_classes']
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.5)
    bboxes = utils.nms(bboxes, 0.45, method='nms')

    res = {}
    objs = []
    for bbox in bboxes:
        objs.append({'x1':bbox[0], 'y1':bbox[1], 'x2':bbox[2], 'y2':bbox[3], 'confidence':bbox[4], 'class':bbox[5]})
    res['objs'] = objs 

    return res



class Item(BaseModel):
    camera_no: str
    image: str

def params_parse(item: Item):
    camera_no = item.camera_no
    image_data = base64.b64decode(item.image)
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
    return camera_no, image


@app.post("/obj_detect_service")
async def obj_detect_service(item: Item):

    try:
        camera_no, image = params_parse(item)
    except:
        logging.error('params_parse error!')
        return -1

    ret = obj_detect(image)

    return ret



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")