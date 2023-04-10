# python dgl_gpu.py --dataset=livejournal
# python dgl_gpu.py --dataset=ogbn-products
# python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu

# python matrix.py --dataset=livejournal
# python matrix.py --dataset=ogbn-products
# python matrix.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python matrix.py --dataset=friendster --use-uva=True --device=cpu

# python matrix_batching.py --dataset=livejournal
# python matrix_batching.py --dataset=ogbn-products
python matrix_batching.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
python matrix_batching.py --dataset=friendster --use-uva=True --device=cpu