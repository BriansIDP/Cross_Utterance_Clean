export CUDA_VISIBLE_DEVICES=0 #${X_SGE_CUDA_DEVICE}
export PATH="/home/miproj/urop.2018/gs534/Software/anaconda3/bin:$PATH"

exp_no=2
model_name="atten"
prev_len=72
post_len=72

FlvmodelPath=${PWD}/models
expdir=${PWD}/${model_name}_${prev_len}_${post_len}_${exp_no}

mkdir -p $expdir

python jointtrain_singleseg.py \
    --cuda \
    --seed 999 \
    --nhid 768 \
    --emsize 256 \
    --lr 10 \
    --FLlr 0.5 \
    --clip 0.25 \
    --FLvclip 0.5 \
    --batchsize 64 \
    --wdecay 2e-6 \
    --bptt 12 \
    --naux 512 \
    --reset 1 \
    --epochs 39 \
    --maxlen_prev $prev_len \
    --maxlen_post $post_len \
    --FLvmodel Flvmodel/model.pt \
    --save ${expdir}/L2model.new_${model_name}.${prev_len}_${post_len}_${exp_no}.pt \
    --FLvsave ${expdir}/L2model.FLv.new_${model_name}.${prev_len}_${post_len}_${exp_no}.pt \
    --logfile ${expdir}/trainlog.new_${model_name}.${prev_len}_${post_len}_${exp_no}.txt \
    --updatedelay 1 \
    --outputcell 1 \
    --useatten \
    --nhead 1 \
    --alpha 0.0000 \
    --use_sampling \
    --errorfile error_sampling/work/confusions.txt \
    --reference error_sampling/train.ref \
    --ratio 5 \
