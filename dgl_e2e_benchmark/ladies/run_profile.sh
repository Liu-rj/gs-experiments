nsys profile -o ladies_dcsr_coo   python3 train_matrix.py --dataset papers100m --device cpu --use-uva --sampler coo
nsys profile -o ladies_dcsr_best   python3 train_matrix.py --dataset papers100m --device cpu --use-uva --sampler best
nsys profile -o ladies_full_coo   python3 train_matrix.py --dataset papers100m --device cpu --use-uva --sampler best