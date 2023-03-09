export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data2/datasets/mid_drone

LOG_PATH4=/home/sj/tmp/mono_drone_cmt_l3_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --cmt_layer=3 --intrinsic_learning

LOG_PATH5=/home/sj/tmp/mono_drone_cmt_l4_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --cmt_layer=4 --intrinsic_learning

LOG_PATH6=/home/sj/tmp/many_drone_cmt_l3_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --batch_size=8 --cmt_layer=3 --intrinsic_learning

LOG_PATH7=/home/sj/tmp/many_drone_cmt_l4_intrinsic
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --batch_size=8 --cmt_layer=4 --intrinsic_learning


LOG_PATH1=/home/sj/tmp/mono_drone_cmt_l3_intrinsic_fpi
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --cmt_layer=3 --intrinsic_learning --freeze_with_pose_intrinsic

LOG_PATH2=/home/sj/tmp/mono_drone_cmt_l4_intrinsic_fpi
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --cmt_layer=4 --intrinsic_learning --freeze_with_pose_intrinsic

LOG_PATH3=/home/sj/tmp/many_drone_cmt_l3_intrinsic_fpi
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --batch_size=8 --cmt_layer=3 --intrinsic_learning --freeze_with_pose_intrinsic

LOG_PATH4=/home/sj/tmp/many_drone_cmt_l4_intrinsic_fpi
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --batch_size=8 --cmt_layer=4 --intrinsic_learning --freeze_with_pose_intrinsic


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --cmt_layer=3

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --cmt_layer=4

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH6/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --cmt_layer=3
 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH7/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --cmt_layer=4
 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --cmt_layer=3

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --cmt_layer=4

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH3/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --cmt_layer=3
 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --cmt_layer=4
 done


