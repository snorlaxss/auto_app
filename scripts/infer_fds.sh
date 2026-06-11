python runners/fds_infer.py \
  --realsense \
  --yolo results/ckpts/YOLO/best.pt \
  --fds FoundationStereo/model_best_bp2_smol.pth \
  --score /home/kewei/repo/GenPose2/results/ckpts/scorenet.pth \
  --scale /home/kewei/repo/GenPose2/results/ckpts/scalenet.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show
