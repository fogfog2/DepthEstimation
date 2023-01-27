export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#ucl
DATA_PATH=/media/sj/data/colon_new2


LOG_PATH1=/home/sj/tmp/mono_cn2_cmt
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_monodepth --no_teacher --png --train_model=cmt --num_epochs=50


LOG_PATH2=/home/sj/tmp/mono_cn2_resnet
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=50


LOG_PATH3=/home/sj/tmp/many_cn2_cmt_r50
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --png --train_model=cmt --num_epochs=50


LOG_PATH4=/home/sj/tmp/many_cn2_resnet_r50
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH4 --png --train_model=resnet --num_epochs=50


LOG_PATH5=/home/sj/tmp/many_cn2_cmt_r50_att
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH5 --png --train_model=cmt --num_epochs=50 --use_attention_decoder

LOG_PATH6=/home/sj/tmp/many_cn2_resnet_r50_att
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH6 --png --train_model=resnet --num_epochs=50 --use_attention_decoder

LOG_PATH7=/home/sj/tmp/mono_cn2_cmt_att
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH7 --train_monodepth --no_teacher --png --train_model=cmt --num_epochs=50 --use_attention_decoder


LOG_PATH8=/home/sj/tmp/mono_cn2_resnet_att
python depthestimation/train.py --data_path=$DATA_PATH --log_dir=$LOG_PATH8 --train_monodepth --no_teacher --png --train_model=resnet --num_epochs=50 --use_attention_decoder

