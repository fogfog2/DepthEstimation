export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/colon_new2


LOG_PATH1=/home/sj/tmp/mono_cn2_resnet_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.5

LOG_PATH2=/home/sj/tmp/mono_cn2_resnet_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.1


LOG_PATH3=/home/sj/tmp/mono_cn2_resnet_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.7


LOG_PATH4=/home/sj/tmp/many_cn2_resnet_r50_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.5


LOG_PATH5=/home/sj/tmp/many_cn2_resnet_r50_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.1

LOG_PATH6=/home/sj/tmp/many_cn2_resnet_r50_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.7


