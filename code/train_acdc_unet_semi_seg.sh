CUDA_VISIBLE_DEVICES=0 python train_MAP_2D.py --root_path ../../data/ACDC --labeled_num 3 --max_iterations 30000 --batch_size 16 --labeled_bs 8 && \
CUDA_VISIBLE_DEVICES=0 python train_MAP_2D.py --root_path ../../data/ACDC --labeled_num 7 --max_iterations 30000 --batch_size 16 --labeled_bs 8 
