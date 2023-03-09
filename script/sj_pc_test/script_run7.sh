export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
#DATA_PATH=/media/sj/data/colon_new2

DATA_PATH=/media/sj/data/datasets/colon_dummy

LOG_PATH4=/home/sj/tmp/many_co_cmt3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --png --train_model=cmt --cmt_layer=3 --num_epochs=30 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --learning_rate=5e-5 --freeze_teacher_epoch=15

LOG_PATH5=/home/sj/tmp/many_co_cmt4
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=cmt --cmt_layer=4 --num_epochs=30 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --learning_rate=5e-5 --freeze_teacher_epoch=15

LOG_PATH6=/home/sj/tmp/many_co_cmt3_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --png --train_model=cmt --cmt_layer=3 --num_epochs=30 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --learning_rate=5e-5 --freeze_teacher_epoch=15

LOG_PATH7=/home/sj/tmp/many_co_cmt4_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --png --train_model=cmt --cmt_layer=4 --num_epochs=30 --scheduler_step_size=5 --scheduler_step_ratio=0.5 --learning_rate=5e-5 --freeze_teacher_epoch=15




