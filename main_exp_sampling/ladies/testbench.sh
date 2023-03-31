# python dgl_gpu.py --dataset=livejournal
# python dgl_gpu.py --dataset=ogbn-products
# python dgl_gpu.py --dataset=ogbn-papers100M --use-uva=True --device=cpu
# python dgl_gpu.py --dataset=friendster --use-uva=True --device=cpu

python author_cpu.py --dataset=livejournal
python author_cpu.py --dataset=ogbn-products
python author_cpu.py --dataset=ogbn-papers100M
python author_cpu.py --dataset=friendster