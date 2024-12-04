GPUS=$1
dataset=$2
multi_scale=$4

if [ $dataset == 0 ];then
  ACDC=('fog' 'night' )
elif [ $dataset == 1 ]; then
    ACDC=('rain' 'snow')
elif [ $dataset == 2 ]; then
  ACDC=('fog' 'night' 'rain' 'snow')
else
  ACDC=($dataset)
fi


echo ${ACDC[*]}

#config_path='configs/tta/train_acdc.py'
config_path='configs/tta/train_acdc.py'

for data in ${ACDC[@]}
do
      export CUDA_VISIBLE_DEVICES=$GPUS &&  CUDA_LAUNCH_BLOCKING=1 && \
      python ./TTA/main_tta.py \
        --domain=$data \
        --config=$config_path \
        --baseline=$3 \
        --tta=$multi_scale

done



# bash TTA/exp/train_acdc.sh 0 fog ModelMerging 0
# bash TTA/exp/train_acdc.sh 1 night ModelMerging 0
# bash TTA/exp/train_acdc.sh 2 rain ModelMerging 0
# bash TTA/exp/train_acdc.sh 3 snow ModelMerging 0