import requests
import json
import time
import base64

params={
    "camera_no": "0",
    "image": ""
}


url='http://127.0.0.1:5000/car_alarm_service'

image_file = 'car.jpg'

with open (image_file,'rb') as f:
    params['image'] = base64.b64encode(f.read()).decode()

# print(params['image'])

time1=time.time()
html = requests.post(url, json.dumps(params))
print('发送post数据请求成功!')
print('返回post结果如下：')
print(html.text)

time2=time.time()
print('总共耗时：' + str(time2 - time1) + 's')