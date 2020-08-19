export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

exp_no=1
model_name="atten"
prev_len=36
post_len=36
seglen=36

FlvmodelPath=${PWD}/models
expdir=${PWD}/${model_name}_${prev_len}_${post_len}_${exp_no}

mkdir -p $expdir

python jointtrain_singleseg.py \
    --cuda \
    --seed 999 \
    --nhid 768 \
    --emsize 256 \
    --lr 10 \
    --FLlr 1.0 \
    --clip 0.25 \
    --FLvclip 2 \
    --batchsize 64 \
    --wdecay 2e-6 \
    --bptt 12 \
    --naux 768 \
    --reset 1 \
    --epochs 30 \
    --maxlen_prev $prev_len \
    --maxlen_post $post_len \
    --seglen ${seglen} \
    --FLvmodel Flvmodel/model.pt \
    --save ${expdir}/L2model.new_${model_name}.${prev_len}_${post_len}_${exp_no}.pt \
    --FLvsave ${expdir}/L2model.FLv.new_${model_name}.${prev_len}_${post_len}_${exp_no}.pt \
    --logfile ${expdir}/trainlog.new_${model_name}.${prev_len}_${post_len}_${exp_no}.txt \
    --updatedelay 1 \
    --outputcell 1 \
    --useatten \
    --nhead 1 \
    --alpha 0.0000 \
