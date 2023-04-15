# python graphsage_matrix.py --dataset=friendster  --data-type=int --use-uva --device=cpu
# python graphsage_matrix.py --dataset=friendster  --data-type=long --use-uva --device=cpu


python pyg_cpu.py --dataset=livejournal
python pyg_cpu.py --dataset=ogbn-products
python pyg_cpu.py --dataset=ogbn-papers100M
python pyg_cpu.py --dataset=friendster
