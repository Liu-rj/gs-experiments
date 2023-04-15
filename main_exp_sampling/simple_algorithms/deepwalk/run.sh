# echo "1====DGL========="
# python deepwalk_dgl.py --batchsize=1024 --dataset=livejournal 
# python deepwalk_dgl.py --batchsize=1024 --dataset=products
# python deepwalk_dgl.py --batchsize=1024 --dataset=papers100m --device=cpu --use-uva
# python deepwalk_dgl.py --batchsize=1024 --dataset=friendster --device=cpu --use-uva
echo "2=====bigbatch=5120======="
python deepwalk_matrix.py --batchsize=1024 --dataset=livejournal --big-batch=51200
python deepwalk_matrix.py --batchsize=1024 --dataset=products --big-batch=51200
python deepwalk_matrix.py --batchsize=1024 --dataset=papers100m --device=cpu --use-uva --big-batch=5120
python deepwalk_matrix.py --batchsize=1024 --dataset=friendster --device=cpu --use-uva --big-batch=2048
