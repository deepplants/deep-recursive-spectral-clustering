OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py \
--arch vit_tiny \
--data_path ../dataset_self_AGM/AGM_09_02_2023_patches \
--batch_size_per_gpu 400 \
--use_fp16 True \
--local_crops_number 8 \
--momentum_teacher 0.995 \
--norm_last_layer False \
--out_dim 16384 \
--output_dir ./checkpoints22 \
--num_workers 32 \
--epochs 501 \
--global_crops_scale 0.7 1.0 \
--local_crops_scale 0.3 0.7 \
# --warmup_teacher_temp_epochs 30 \
# --freeze_last_layer 1 \
# --teacher_temp 0.07 \
# --lr 0.0005 \
# --warmup_epochs 5 \
