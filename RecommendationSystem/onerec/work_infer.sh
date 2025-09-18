
cd /home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec
# MASTER_PORT=29513 python train_onerec.py > /home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/exp/toy/dual_tower_model/log.txt 2>&1 &
MASTER_PORT=29532 CUDA_VISIBLE_DEVICES=0 python infer.py > /home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/exp/art/dual_tower_model/version_0/test_log.txt 2>&1 &
