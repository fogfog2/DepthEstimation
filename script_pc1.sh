export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon

LOG_PATH1=/home/sj/tmp/mono_drl_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=1.0

LOG_PATH2=/home/sj/tmp/mono_drl_05
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.5

LOG_PATH3=/home/sj/tmp/mono_drl_01
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.1

LOG_PATH4=/home/sj/tmp/mono_drl_05_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.5 --train_model=cmt

LOG_PATH5=/home/sj/tmp/mono_drl_1_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=1.0 --train_model=cmt

LOG_PATH6=/home/sj/tmp/mono_drl_1_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --depth_reconstruction_loss --reconstruction_loss_weight=0.1 --train_model=cmt

#LOG_PATH5=/home/sj/tmp/mono_adam_s5_r05_reprojection_loss
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --use_adamw --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5






