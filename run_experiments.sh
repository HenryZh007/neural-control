#!/bin/bash

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting experiment batch run..."
echo "=========================================="

# # 1. any_node_cem.py
# echo ""
# echo "[1/7] Running any_node_cem.py..."
# echo "=========================================="
# python3 final_exp/any_node_cem.py
# echo "Finished any_node_cem.py with exit code: $?"

# # 2. letter_curve_cem.py
# echo ""
# echo "[2/7] Running letter_curve_cem.py..."
# echo "=========================================="
# python3 final_exp/letter_curve_cem.py
# echo "Finished letter_curve_cem.py with exit code: $?"

# # 3. letter_curve_MPC.py
# echo ""
# echo "[3/7] Running letter_curve_MPC.py..."
# echo "=========================================="
# python3 final_exp/letter_curve_MPC.py
# echo "Finished letter_curve_MPC.py with exit code: $?"

# # 4. letter_curve_noMPC.py
# echo ""
# echo "[4/7] Running letter_curve_noMPC.py..."
# echo "=========================================="
# python3 final_exp/letter_curve_noMPC.py
# echo "Finished letter_curve_noMPC.py with exit code: $?"

# # 5. letter_curve_spsa.py
# echo ""
# echo "[5/7] Running letter_curve_spsa.py..."
# echo "=========================================="
# python3 final_exp/letter_curve_spsa.py
# echo "Finished letter_curve_spsa.py with exit code: $?"

# # 6. middle_tracking_cem.py
# echo ""
# echo "[6/7] Running middle_tracking_cem.py..."
# echo "=========================================="
# python3 final_exp/middle_tracking_cem.py
# echo "Finished middle_tracking_cem.py with exit code: $?"

# # 7. middle_tracking_spsa.py
# echo ""
# echo "[7/7] Running middle_tracking_spsa.py..."
# echo "=========================================="
# python3 final_exp/middle_tracking_spsa.py
# echo "Finished middle_tracking_spsa.py with exit code: $?"

# # 1. any_node_icem.py
# echo ""
# echo "[1/3] Running any_node_icem.py..."
# echo "=========================================="
# python3 final_exp/any_node_icem.py
# echo "Finished any_node_icem.py with exit code: $?"

# # 2. middle_tracking_icem.py
# echo ""
# echo "[2/3] Running middle_tracking_icem.py..."
# echo "=========================================="
# python3 final_exp/middle_tracking_icem.py
# echo "Finished middle_tracking_icem.py with exit code: $?"

# # 3. letter_curve_icem.py
# echo ""
# echo "[3/3] Running letter_curve_icem.py..."
# echo "=========================================="
# python3 final_exp/letter_curve_icem.py
# echo "Finished letter_curve_icem.py with exit code: $?"

# 1. any_node_cem.py
echo ""
echo "[1/3] Running any_node_cem.py..."
echo "=========================================="
python3 final_exp/any_node_cem.py
echo "Finished any_node_cem.py with exit code: $?"

# 2. middle_tracking_cem.py
echo ""
echo "[2/3] Running middle_tracking_cem.py..."
echo "=========================================="
python3 final_exp/middle_tracking_cem.py
echo "Finished middle_tracking_cem.py with exit code: $?"

# 3. letter_curve_cem.py
echo ""
echo "[3/3] Running letter_curve_cem.py..."
echo "=========================================="
python3 final_exp/letter_curve_cem.py
echo "Finished letter_curve_cem.py with exit code: $?"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
