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
from car_detect import CarDetect, car_alarm, get_files


app = FastAPI()

with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

alarm_list = [0]*cfg['alarm_range']
car_detect = CarDetect(cfg)
mask_path_list, mask_name_list = get_files('mask/' + cfg['camera_no'])
alarm_list = [0]*cfg['alarm_range']



class Item(BaseModel):
    camera_no: str
    image: str

def params_parse(item: Item):
    camera_no = item.camera_no
    image_data = base64.b64decode(item.image)
    image = cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
    return camera_no, image


@app.post("/car_alarm_service")
async def car_alarm_service(item: Item):

    try:
        camera_no, image = params_parse(item)
    except:
        logging.error('params_parse error!')
        return -1

    ret = car_alarm(image, car_detect, cfg, mask_path_list, alarm_list)
    

    return ret



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")