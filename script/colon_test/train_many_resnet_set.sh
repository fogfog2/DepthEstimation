#edit your path
export PYTHONPATH="${PYTHONPATH}:/home/sj/src/DepthEstimation" 

#data path (# Parent folder of "/images/sequenceNN" path  )
DATA_PATH=/media/sj/data/datasets/colon_new2

#log path 

LOG_PATH1=/home/sj/tmp/many_resnet101_50epoch
LOG_PATH2=/home/sj/tmp/many_resnet50_50epoch
LOG_PATH3=/home/sj/tmp/many_resnet18_50epoch

#run script r101
python ../../depthestimation/train_dummy.py --data_path=$DATA_PATH --log_dir=$LOG_PATH1 --train_model=resnet --num_layers=101 --num_epochs=50 --batch_size=4

#run script r50
python ../../depthestimation/train_dummy.py --data_path=$DATA_PATH --log_dir=$LOG_PATH2 --train_model=resnet --num_layers=50 --num_epochs=50 --batch_size=4

#run script r18
python ../../depthestimation/train_dummy.py --data_path=$DATA_PATH --log_dir=$LOG_PATH3 --train_model=resnet --num_layers=18 --num_epochs=50


