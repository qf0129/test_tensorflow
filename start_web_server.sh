# /usr/bin

 nohup /home/ubuntu/work/test_tensorflow/venv/bin/python3.8 /home/ubuntu/work/test_tensorflow/venv/bin/gunicorn -w 4 -b 0.0.0.0:5001 web_server:app >>web_server.out &
