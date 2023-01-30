export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/datasets/mid_drone

LOG_PATH5=/home/sj/tmp/mono_drone_cmtat_l4

for i in 20 25 29
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH5/mdp/models/weights_$i --eval_split=custom_mid --eval_mono --no_teacher --png --train_model=cmt --use_attention_decoder --cmt_layer=4

 done
