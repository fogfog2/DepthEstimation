export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
#DATA_PATH=/media/sj/data/colon_new2



DATA_PATH=/media/sj/data/datasets/colon_new2

LOG_PATH0=/home/sj/tmp/mono_cn2_res_1

python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH0 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=5e-5 --split=custom_dummy_new --train_monodepth --no_teacher

LOG_PATH1=/home/sj/tmp/mono_cn2_res_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=1e-4 --split=custom_dummy_new --train_monodepth --no_teacher
 
LOG_PATH2=/home/sj/tmp/mono_cn2_res_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=1e-5 --split=custom_dummy_new --train_monodepth --no_teacher

LOG_PATH3=/home/sj/tmp/mono_cn2_res_4
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=2e-5 --split=custom_dummy_new --train_monodepth --no_teacher

DATA_PATH=/media/sj/data/datasets/colon_dummy

LOG_PATH8=/home/sj/tmp/many_co_res_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH8 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --batch_size=8 --learning_rate=1e-5

LOG_PATH9=/home/sj/tmp/many_co_res_4
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH9 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --batch_size=8 --learning_rate=2e-5


LOG_PATH4=/home/sj/tmp/monoco_res_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --batch_size=8 --learning_rate=5e-5 --train_monodepth --no_teacher

LOG_PATH5=/home/sj/tmp/mono_co_res_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=1e-4 --train_monodepth --no_teacher

LOG_PATH6=/home/sj/tmp/mono_co_res_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=1e-5 --train_monodepth --no_teacher

LOG_PATH7=/home/sj/tmp/mono_co_res_4
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --png --train_model=resnet --num_epochs=50 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --learning_rate=2e-5 --train_monodepth --no_teacher




