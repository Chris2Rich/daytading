#!/bin/bash
cd /root/daytading
git pull origin main
/root/venv/bin/pip install -r daytading/requirements.txt
systemctl restart daytading
