export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/colon_new2


#LOG_PATH1=/home/sj/tmp/mono_cn2_resnet50_1
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --png --#train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.5 --num_layers=50

#fail
#LOG_PATH5=/home/sj/tmp/many_cn2_resnet50_r50_1
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=resnet --num_epochs=100 --#scheduler_step_size=10 --scheduler_step_ratio=0.5 --num_layers=50


LOG_PATH2=/home/sj/tmp/mono_cn2_resnet50_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.1 --num_layers=50


LOG_PATH3=/home/sj/tmp/mono_cn2_resnet50_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=100 --scheduler_step_size=10 --scheduler_step_ratio=0.7 --num_layers=50

