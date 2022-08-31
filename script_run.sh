export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon

LOG_PATH5=/home/sj/tmp/mono_drl_01_adam
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.1

LOG_PATH6=/home/sj/tmp/mono_drl_01_adam_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --train_monodepth --no_teacher --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.1 --train_model=cmt

