#!/usr/bin/env bash
nohup python3 obj_detect_service.py &
nohup python3 alarm_decision_service.py &

echo "service start success!"


