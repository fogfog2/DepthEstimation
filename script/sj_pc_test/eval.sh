export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
#DATA_PATH=/home/sj/colon
DATA_PATH=/media/sj/data/mid_drone

LOG_PATH1=/home/sj/tmp/mono_drone_resnet_r05_1
LOG_PATH2=/home/sj/tmp/mono_drone_cmt_r05_1

SET=$(seq 20 39)

for i in $SET
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono
 done

for i in $SET
 do 
  python depthestimation/evaluate_depth_mid.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH2/mdp/models/weights_$i --train_monodepth --no_teacher --eval_split=custom_mid --png --eval_mono --train_model=cmt
 done
