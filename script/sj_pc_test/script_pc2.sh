export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
#DATA_PATH=/home/sj/colon

#mid_drone
DATA_PATH=/media/sj/data/mid_drone

#LOG_PATH1=/home/sj/tmp/mono_drone_resnet_r05_1
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5

LOG_PATH2=/home/sj/tmp/mono_drone_cmt_r01_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --train_model=cmt

SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono --train_model=cmt
 done

#LOG_PATH3=/home/sj/tmp/mono_drone_resnet_r05_adamw
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --use_adamw --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5

#LOG_PATH4=/home/sj/tmp/mono_drone_cmt_r05_adamw
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --use_adamw --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --train_model=cmt


#LOG_PATH1=/home/sj/tmp/mono_drone_resnet_r05_1/mdp/models/weights_39
#python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1 --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono

#LOG_PATH2=/home/sj/tmp/mono_drone_cmt_r05_1/mdp/models/weights_39
#python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2 --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono --train_model=cmt


#LOG_PATH3=/home/sj/tmp/mono_drone_resnet_r05_adamw/mdp/models/weights_39
#python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH3 --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono

#LOG_PATH4=/home/sj/tmp/mono_drone_cmt_r05_adamw/mdp/models/weights_39
#python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4 --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono --train_model=cmt

