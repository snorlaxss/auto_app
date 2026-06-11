python runners/rosbox.py \
  --realsense \
  --yolo results/ckpts/YOLO/best.pt \
  --score /home/kewei/repo/GenPose2/results/ckpts/scorenet.pth \
  --scale /home/kewei/repo/GenPose2/results/ckpts/scalenet.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show
