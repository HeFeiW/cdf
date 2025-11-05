#!/bin/bash
# Training script for all 4 fingers of LeapHand with base DoF
# Each finger has 6DoF base + 4DoF joints = 10DoF total

echo "Training LeapHand CDF models with base DoF..."
echo "Network type: ${NETWORK_TYPE:-mlp}"
echo "Epochs: ${EPOCHS:-50000}"

NETWORK_TYPE=${NETWORK_TYPE:-mlp}
EPOCHS=${EPOCHS:-50000}
BATCH_X=${BATCH_X:-10}
BATCH_Q=${BATCH_Q:-100}
MAX_Q_PER_LINK=${MAX_Q_PER_LINK:-100}

# Train finger 0
echo "==================================="
echo "Training Finger 0 (Index finger)..."
echo "==================================="
# 打印将要执行的命令
echo "Executing command:"
echo "python3 para_nn_cdf_v2.py --train --use_base --robot leaphand --serial_idx 0 --network_type $NETWORK_TYPE --data_path data_with_base_dof_0.pt --model_dict leaphand_finger0_${NETWORK_TYPE}_base.pt --epoches $EPOCHS --batch_x $BATCH_X --batch_q $BATCH_Q --max_q_per_link $MAX_Q_PER_LINK --with_writer"


python3 para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 0 \
    --network_type $NETWORK_TYPE \
    --data_path data_with_base_dof_0.npy \
    --model_dict leaphand_finger0_${NETWORK_TYPE}_base.pt \
    --epoches $EPOCHS \
    --batch_x $BATCH_X \
    --batch_q $BATCH_Q \
    --max_q_per_link $MAX_Q_PER_LINK \
    --with_writer

# Train finger 1
echo "==================================="
echo "Training Finger 1 (Middle finger)..."
echo "==================================="
python3 para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 1 \
    --network_type $NETWORK_TYPE \
    --data_path data_with_base_dof_1.npy \
    --model_dict leaphand_finger1_${NETWORK_TYPE}_base.pt \
    --epoches $EPOCHS \
    --batch_x $BATCH_X \
    --batch_q $BATCH_Q \
    --max_q_per_link $MAX_Q_PER_LINK \
    --with_writer

# Train finger 2
echo "==================================="
echo "Training Finger 2 (Ring finger)..."
echo "==================================="
python3 para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 2 \
    --network_type $NETWORK_TYPE \
    --data_path data_with_base_dof_2.npy \
    --model_dict leaphand_finger2_${NETWORK_TYPE}_base.pt \
    --epoches $EPOCHS \
    --batch_x $BATCH_X \
    --batch_q $BATCH_Q \
    --max_q_per_link $MAX_Q_PER_LINK \
    --with_writer

# Train finger 3
echo "==================================="
echo "Training Finger 3 (Thumb)..."
echo "==================================="
python3 para_nn_cdf_v2.py \
    --train \
    --use_base \
    --robot leaphand \
    --serial_idx 3 \
    --network_type $NETWORK_TYPE \
    --data_path data_with_base_dof_3.npy \
    --model_dict leaphand_finger3_${NETWORK_TYPE}_base.pt \
    --epoches $EPOCHS \
    --batch_x $BATCH_X \
    --batch_q $BATCH_Q \
    --max_q_per_link $MAX_Q_PER_LINK \
    --with_writer

echo "==================================="
echo "All fingers trained successfully!"
echo "==================================="
