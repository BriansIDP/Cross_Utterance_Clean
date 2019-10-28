# Cross_Utterance_Clean
This directory contains different LSTM-based cross-utterance language models (LMs)

-------------------------------------------------------------------------------
1. Train/pre-train the first-level LM using vanillaLSTM.sh
   The model will be saved in Flvmodel directory
2. Train the cross-utterance LM using jointatten.sh
   The model will be saved in atten_<prev>_<post>_<exp_no> directory

