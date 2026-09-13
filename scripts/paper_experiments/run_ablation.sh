#!/bin/bash
OUT_DIRS="/home/ycl/AICO-Intellig/results/ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148 /home/ycl/AICO-Intellig/results/ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116 /home/ycl/AICO-Intellig/results/ex_heat_time_v1_full_200_m100_t3600_r1_ /home/ycl/AICO-Intellig/results/heat_v1_full_180_m30_t90_r3 /home/ycl/AICO-Intellig/results/iaea_v1_full_180_m30_t90_r3_20260708_102705 /home/ycl/AICO-Intellig/results/sns_v1_full_150_m40_t90_r4_20260429_234621 /home/ycl/AICO-Intellig/results/sns_v3_full_240_m30_t90_r3_more1 /home/ycl/AICO-Intellig/results/thmf_v1_full_180_m30_t90_r3_20260708_102705"

echo "Step | Mean AUC Gain | Win Rate | Switch Rate"
echo "-----------------------------------------------"

for STEPS in 5 10 15 20 25 30; do
    python /home/ycl/AICO-Intellig/scripts/train_curve_policy.py \
        --out_dirs $OUT_DIRS \
        --decision_eval $STEPS \
        --switch_threshold 0.5 \
        > /tmp/ablation_${STEPS}.log 2>&1
    
    MEAN_GAIN=$(grep -A 2 '"auc_gain_vs_default"' /tmp/ablation_${STEPS}.log | grep '"mean"' | head -n 1 | awk -F: '{print $2}' | tr -d ' ,')
    SWITCH_RATE=$(grep '"switch_rate"' /tmp/ablation_${STEPS}.log | head -n 1 | awk -F: '{print $2}' | tr -d ' ,')
    WIN_RATE=$(grep '"win_rate_vs_default_auc"' /tmp/ablation_${STEPS}.log | head -n 1 | awk -F: '{print $2}' | tr -d ' ,')
    
    printf "%4s | %13s | %8.2f%% | %10.2f%%\n" "$STEPS" "$MEAN_GAIN" $(echo "$WIN_RATE * 100" | bc -l) $(echo "$SWITCH_RATE * 100" | bc -l)
done
