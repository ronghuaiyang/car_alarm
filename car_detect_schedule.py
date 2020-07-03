# -*- coding: utf-8 -*-
import sys
import os

from apscheduler.schedulers.blocking import BlockingScheduler
import yaml

from car_decect import CarDetect, car_alarm, get_files
# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
# from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_EXECUTED



if __name__ == "__main__":

    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    alarm_list = [0]*cfg['alarm_range']
    
    car_detect = CarDetect(cfg)
    mask_path_list, mask_name_list = get_files('mask/' + cfg['camera_no'])
    scheduler = BlockingScheduler()

    scheduler.add_job(car_alarm, 'interval', args=[car_detect, cfg, mask_path_list, alarm_list], seconds=cfg['detect_interval'])

    scheduler.start()
