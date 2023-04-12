# python matrix_batching.py --dataset=livejournal
# python matrix_batching.py --dataset=ogbn-products
# python matrix_batching.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python matrix_batching.py --dataset=friendster --use-uva=True --device=cpu

# python dgl_gpu.py --dataset=livejournal
# python dgl_gpu.py --dataset=ogbn-products
# python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu

python pyg_mix.py --dataset=livejournal
python pyg_mix.py --dataset=ogbn-products
# python pyg_mix.py --dataset=ogbn-papers100M --device=cpu
# python pyg_mix.py --dataset=friendster --device=cpu