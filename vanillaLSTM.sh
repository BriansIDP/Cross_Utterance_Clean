export CUDA_VISIBLE_DEVICES=0 #${X_SGE_CUDA_DEVICE}
export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"

FlvDir=${PWD}/Flvmodel
mkdir $FlvDir

python train_with_dataloader.py \
    --data ${PWD}/data/AMI/ \
    --cuda \
    --emsize 256 \
    --nhid 768 \
    --dropout 0.5 \
    --rnndrop 0.25 \
    --epochs 2 \
    --lr 10 \
    --clip 0.25 \
    --nlayers 1 \
    --batch_size 32 \
    --bptt 36 \
    --wdecay 5e-6 \
    --model LSTM \
    --reset 1 \
    --logfile ${FlvDir}/vanillaLSTM.log \
    --save ${FlvDir}/model.pt

