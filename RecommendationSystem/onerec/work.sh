
cd /home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec
MASTER_PORT=29515 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_onerec.py > /home/tangzixuan.8/Generative_recall/generative_recall_rank/compare_exp/onerec/exp/video_game/dual_tower_model/log.txt 2>&1 &