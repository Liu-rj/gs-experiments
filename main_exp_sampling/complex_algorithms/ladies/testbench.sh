# python matrix.py --dataset=livejournal
# python matrix.py --dataset=ogbn-products
# python matrix.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python matrix.py --dataset=friendster --use-uva=True --device=cpu

# python dgl_gpu.py --dataset=livejournal
# python dgl_gpu.py --dataset=ogbn-products
# python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu

# python author_cpu.py --dataset=livejournal
# python author_cpu.py --dataset=ogbn-products
# python author_cpu.py --dataset=ogbn-papers100M
# python author_cpu.py --dataset=friendster

python dgl_gpu.py --dataset=livejournal --device=cpu
python dgl_gpu.py --dataset=ogbn-products --device=cpu
python dgl_gpu.py --dataset=ogbn-papers100M --device=cpu
python dgl_gpu.py --dataset=friendster --device=cpu