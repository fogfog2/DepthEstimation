export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/datasets/mid_drone

LOG_PATH4=/media/sj/data/result_models/mid_drone_test/mono_drone_cmtat_0


LOG_PATH5=/media/sj/data/result_models/mid_drone_test/mono_drone_cmtat_1


LOG_PATH6=/media/sj/data/result_models/mid_drone_test/many_drone_cmtat_2

LOG_PATH7=/media/sj/data/result_models/mid_drone_test/many_drone_cmtat_3

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

