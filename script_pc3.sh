export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
#DATA_PATH=/home/sj/colon

#kitti
DATA_PATH=/home/sj/kitti

LOG_PATH1=/home/sj/tmp/mono_kitti_resnet_r05
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --use_adamw --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5

LOG_PATH2=/home/sj/tmp/mono_kitti_cmt_r05
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --use_adamw --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --train_model=cmt


LOG_PATH3=/home/sj/tmp/many_kitti_resnet_r01
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --use_adamw --freeze_teacher_epoch=15 --png 

LOG_PATH4=/home/sj/tmp/many_kitti_cmt_r01
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --use_adamw --freeze_teacher_epoch=15 --png --train_model=cmt


