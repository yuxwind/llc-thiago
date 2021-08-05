ps aux|grep todo|awk '{print $2}'|xargs -i kill -9 {}
ps aux|grep get_activation_patterns.py|awk '{print $2}'|xargs -i kill -9 {}
ps aux|grep get_activation_patterns.py
