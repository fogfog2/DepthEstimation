export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon

LOG_PATH0=/home/sj/tmp/mono_drl_07_adam
#python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH0 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.7

for i in 20 30 39
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH0/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done


LOG_PATH1=/home/sj/tmp/mono_drl_07_adam_nossim_01_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.7 --depth_reconstruction_loss --reconstruction_loss_weight=0.1 --train_model=cmt
for i in 20 30 39
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png --train_model=cmt
 done

LOG_PATH2=/home/sj/tmp/mono_drl_03_adam_nossim_01_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.3 --depth_reconstruction_loss --reconstruction_loss_weight=0.1 --train_model=cmt

for i in 20 30 39
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png --train_model=cmt
 done



LOG_PATH5=/home/sj/tmp/mono_drl_07_adam_nossim_05_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.7 --depth_reconstruction_loss --reconstruction_loss_weight=0.5 --train_model=cmt

for i in 20 30 39
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png --train_model=cmt
 done


