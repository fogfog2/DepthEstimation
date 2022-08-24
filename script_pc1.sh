export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon

LOG_PATH1=/home/sj/tmp/mono_s5_r09
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.9

LOG_PATH2=/home/sj/tmp/mono_s5_r07
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.7

LOG_PATH3=/home/sj/tmp/mono_s5_r03
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_monodepth --no_teacher --use_adamw --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.3

LOG_PATH4=/home/sj/tmp/mono_adam_s5_r05_adam
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --no_teacher --no_reprojection_loss_mask --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5

LOG_PATH5=/home/sj/tmp/mono_adam_s5_r05_reprojection_loss
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --use_adamw --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5


SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done


SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done



SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH3/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done



SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done



SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done


