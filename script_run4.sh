export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/colon_new2




LOG_PATH4=/home/sj/tmp/many_cn2_res_hr_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --height=384 --width=512 --batch_size=2

LOG_PATH5=/home/sj/tmp/many_cn2_res_hr_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.1 --height=384 --width=512 --batch_size=2

LOG_PATH6=/home/sj/tmp/many_cn2_res_hr_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.7 --height=384 --width=512 --batch_size=2

LOG_PATH1=/home/sj/tmp/mono_cn2_res_hr_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --height=384 --width=512 --batch_size=4

LOG_PATH2=/home/sj/tmp/mono_cn2_res_hr_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.1 --height=384 --width=512 --batch_size=4

LOG_PATH3=/home/sj/tmp/mono_cn2_res_hr_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=50 --scheduler_step_size=5 --scheduler_step_ratio=0.7 --height=384 --width=512 --batch_size=4
