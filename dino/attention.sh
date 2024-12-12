CUDA_VISIBLE_DEVICES=-1 python visualize_attention.py \
--arch vit_tiny \
--patch_size 16 \
--pretrained_weights dino/checkpoints17/checkpoint0500.pth \
--image_path ../dataset_self_AGM/val/specie_174_stressed/MLZ1_171628_171636_MLZ1_C1_L10_T2_Tr7_1665031177_0_.jpg \
--output_dir attentions \
--threshold 0.2