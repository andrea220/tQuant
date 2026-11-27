#ps aux | grep tqstudio
cd /home/quants/tQuant/tQuant
source ../.venv/bin/activate
export PYTHONPATH=/home/quants/tQuant/tQuant
nohup python tqstudio &