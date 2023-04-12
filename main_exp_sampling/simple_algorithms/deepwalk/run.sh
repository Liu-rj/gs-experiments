echo "1============"
python deepwalk_dgl.py --dataset=livejournal
python deepwalk_dgl.py --dataset=products
python deepwalk_dgl.py --dataset=friendster --device=cpu --use-uva
python deepwalk_dgl.py --dataset=papers100m --device=cpu --use-uva
echo "2============"
python deepwalk_matrix_nobatching.py --dataset=livejournal
python deepwalk_matrix_nobatching.py --dataset=products
python deepwalk_matrix_nobatching.py --dataset=friendster --device=cpu --use-uva
python deepwalk_matrix_nobatching.py --dataset=papers100m --device=cpu --use-uva
echo "3============"
python deepwalk_matrix.py --dataset=livejournal --big-batch=640
python deepwalk_matrix.py --dataset=products --big-batch=640
python deepwalk_matrix.py --dataset=friendster --device=cpu --use-uva --big-batch=640
python deepwalk_matrix.py --dataset=papers100m --device=cpu --use-uva --big-batch=640
echo "4============"
python deepwalk_matrix.py --dataset=livejournal --big-batch=1280
python deepwalk_matrix.py --dataset=products --big-batch=1280
python deepwalk_matrix.py --dataset=friendster --device=cpu --use-uva --big-batch=1280
python deepwalk_matrix.py --dataset=papers100m --device=cpu --use-uva --big-batch=1280
echo "5============"
python deepwalk_matrix.py --dataset=livejournal --big-batch=5120
python deepwalk_matrix.py --dataset=products --big-batch=5120
python deepwalk_matrix.py --dataset=friendster --device=cpu --use-uva --big-batch=5120
python deepwalk_matrix.py --dataset=papers100m --device=cpu --use-uva --big-batch=5120
echo "6============"
python deepwalk_matrix.py --dataset=livejournal --big-batch=10240
python deepwalk_matrix.py --dataset=products --big-batch=10240
python deepwalk_matrix.py --dataset=friendster --device=cpu --use-uva --big-batch=10240
python deepwalk_matrix.py --dataset=papers100m --device=cpu --use-uva --big-batch=10240
echo "7============"
python deepwalk_matrix.py --dataset=livejournal --big-batch=51200
python deepwalk_matrix.py --dataset=products --big-batch=51200
python deepwalk_matrix.py --dataset=friendster --device=cpu --use-uva --big-batch=51200
python deepwalk_matrix.py --dataset=papers100m --device=cpu --use-uva --big-batch=51200
