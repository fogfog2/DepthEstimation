export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/mid_drone

LOG_PATH4=/home/sj/tmp/mono_drone_cmtat_0
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --use_attention_decoder

LOG_PATH5=/home/sj/tmp/mono_drone_cmtat_1
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --train_monodepth --no_teacher --freeze_teacher_epoch=-1 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt --use_attention_decoder

LOG_PATH6=/home/sj/tmp/many_drone_cmtat_2
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.1 --scheduler_step_size=5 --train_model=cmt --use_attention_decoder --batch_size=8

LOG_PATH7=/home/sj/tmp/many_drone_cmtat_3
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --freeze_teacher_epoch=15 --png --scheduler_step_ratio=0.5 --scheduler_step_size=5 --train_model=cmt --use_attention_decoder --batch_size=8

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH4/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --use_attention_decoder

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --use_attention_decoder

 done

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH6/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --use_attention_decoder

 done


for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH7/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --png --train_model=cmt --use_attention_decoder
 done

