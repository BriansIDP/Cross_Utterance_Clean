export CUDA_VISIBLE_DEVICES=0 #${X_SGE_CUDA_DEVICE}
export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

exp_no=$1
max=$2
set=eval
seg=36

python jointforward.py \
    --nbest nbest/time_sorted_${set}.nbestlist \
    --model atten_${max}_${max}_${exp_no}/L2model.new_atten.${max}_${max}_${exp_no}.pt \
    --FLvmodel atten_${max}_${max}_${exp_no}/L2model.FLv.new_atten.${max}_${max}_${exp_no}.pt \
    --lm atten_error_${exp_no} \
    --rnnscale 10 \
    --arrange atten_shared \
    --maxlen ${max} \
    --seglen ${seg} \
    --cuda \
    --ngram nbest/time_sorted_Ldev.nbestlist \
    --logfile nbest/logAtten.txt \
    --outputcell 1 \
    --gscale 12.0 \
    --factor 0.2 \
    --map nbest/${set}.map
