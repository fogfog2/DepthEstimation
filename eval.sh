export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/home/sj/colon


LOG_PATH3=/home/sj/tmp/mono_drl_01_adam


SET=$(seq 20 39)
for i in 20 30 39
 do 
  python depthestimation/evaluate_depth_ucl.py --data_path=$DATA_PATH --load_weights_folder=$LOG_PATH3/mdp/models/weights_$i --eval_split=custom_ucl --eval_mono --no_teacher --png
 done


