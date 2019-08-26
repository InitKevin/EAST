#!/bin/bash
mkdir -p logs/server
gunicorn \
    -w 3 \
    server:app \
    -b 0.0.0.0:9876 -t 120 \
	--error-logfile  logs/server/error.log \
	--access-logfile logs/server/access.log
