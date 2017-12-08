#!/usr/bin/env bash
rm *.bin
rm model*

# Generate libffm files
python generate_libffm_files.py
python generate_oof_libffm_files.py

# Train on folds
./ffm-train -t 11 -r .1 -l 0.000002 -p val_ffm_1.txt trn_ffm_1.txt
./ffm-predict val_ffm_1.txt trn_ffm_1.txt.model val_ffm_1_preds.txt
#
./ffm-train -t 11 -r .1 -l 0.000002 -p val_ffm_2.txt trn_ffm_2.txt
./ffm-predict val_ffm_2.txt trn_ffm_2.txt.model val_ffm_2_preds.txt
#
./ffm-train -t 11 -r .1 -l 0.000002 -p val_ffm_3.txt trn_ffm_3.txt
./ffm-predict val_ffm_3.txt trn_ffm_3.txt.model val_ffm_3_preds.txt
#
./ffm-train -t 11 -r .1 -l 0.000002 -p val_ffm_4.txt trn_ffm_4.txt
./ffm-predict val_ffm_4.txt trn_ffm_4.txt.model val_ffm_4_preds.txt
#
./ffm-train -t 11 -r .1 -l 0.000002 -p val_ffm_5.txt trn_ffm_5.txt
./ffm-predict val_ffm_5.txt trn_ffm_5.txt.model val_ffm_5_preds.txt
# Fit on full train dataset and predict test dataset
./ffm-train -t 11 -r .1 -l 0.000002 alltrainffm.txt
./ffm-predict alltestffm.txt alltrainffm.txt.model output.txt

# Create OOF and submission files
python create_oof_sub.py







