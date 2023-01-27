export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/mid_drone

LOG_PATH0=/home/sj/tmp/mono_drone_resnet_0
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH0 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=resnet

LOG_PATH1=/home/sj/tmp/mono_drone_resnet_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=resnet

LOG_PATH2=/home/sj/tmp/many_drone_resnet_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=resnet

LOG_PATH3=/home/sj/tmp/many_drone_resnet_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=resnet

LOG_PATH4=/home/sj/tmp/mono_drone_cmt_0
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt

LOG_PATH5=/home/sj/tmp/mono_drone_cmt_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt

LOG_PATH6=/home/sj/tmp/many_drone_cmt_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt

LOG_PATH7=/home/sj/tmp/many_drone_cmt_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt


LOG_PATH8=/home/sj/tmp/mono_drone_cmt_4
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH8 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --depth_reconstruction_loss --reconstruction_loss_weight=0.1

LOG_PATH9=/home/sj/tmp/mono_drone_cmt_5
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH9 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt --depth_reconstruction_loss --reconstruction_loss_weight=0.1

LOG_PATH10=/home/sj/tmp/many_drone_cmt_6
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH10 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --depth_reconstruction_loss --reconstruction_loss_weight=0.1

LOG_PATH11=/home/sj/tmp/many_drone_cmt_7
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH11 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt --depth_reconstruction_loss --reconstruction_loss_weight=0.1



for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH0/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=resnet

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=resnet

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=resnet

 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH3/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=resnet
 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH6/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt

 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH7/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt
 done



for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH8/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH9/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH10/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt

 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH11/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt
 done

