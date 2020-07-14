import requests
import json
import time
import base64

params1={
    "camera_no": "01",
    "image": ""
}

params2={
    "camera_no": "01",
    "objs": []
}


# url='http://127.0.0.1:5000/car_alarm_service'
url1='http://127.0.0.1:5000/obj_detect_service'
url2='http://127.0.0.1:5001/alarm_decision_service'


image_file = 'car.jpg'

with open (image_file,'rb') as f:
    params1['image'] = base64.b64encode(f.read()).decode()

# print(params['image'])

time1=time.time()
html = requests.post(url1, json.dumps(params1))
print('发送post数据请求成功!')
print('返回post结果如下：')
print(html.text)
time2=time.time()
print('总共耗时：' + str(time2 - time1) + 's')

time1=time.time()
params2['objs'] = json.loads(html.text)['objs']
html = requests.post(url2, json.dumps(params2))
print('发送post数据请求成功!')
print('返回post结果如下：')
print(html.text)
time2=time.time()
print('总共耗时：' + str(time2 - time1) + 's')
