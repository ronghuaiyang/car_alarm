#!/usr/bin/env bash
# nohup python car_detect_schedule.py >> log.out 2>&1 &
# gunicorn car_alarm_service:app -b 127.0.0.1:5000  -w 1 -k uvicorn.workers.UvicornH11Worker --daemon
# nohup python3 car_alarm_service.py >> log.out 2>&1 &
python3 car_alarm_service.py

