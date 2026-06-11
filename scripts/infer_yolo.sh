python runners/yomni_infer.py \
  --realsense \
  --yolo results/ckpts/YOLO/best.pt \
  --score /home/kewei/repo/GenPose2/results/ckpts/ScoreNet/740boom/ckpt_epoch700.pth \
  --scale /home/kewei/repo/GenPose2/results/ckpts/ScaleNet/ckpt_epoch100.pth \
  --device cuda \
  --frame_gap_threshold 10 \
  --show
