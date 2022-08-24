export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon
LOG_PATH1=/home/sj/tmp/mono_s5_r05_rp_adam


SET=$(seq 20 39)
for i in $SET
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH1/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png 
 done

