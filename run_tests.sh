#!/bin/bash

# NVLink GEMM性能测试脚本
# 测试不同的矩阵存储位置配置

echo "======================================"
echo "NVLink GEMM 性能测试套件"
echo "======================================"
echo ""

# 矩阵尺寸
M=2048
N=2048
K=2048

echo "使用矩阵尺寸: ${M}x${K} × ${K}x${N}"
echo ""

# 测试场景1: 所有数据都在计算GPU上（基准测试）
echo "【场景1】所有数据在本地 - GPU 0计算, A和B都在GPU 0"
./nvlink_gemm -c 0 -a 0 -b 0 -m $M -n $N -k $K
echo ""
echo "======================================"
echo ""

# 测试场景2: 矩阵A在远程GPU
echo "【场景2】矩阵A在远程 - GPU 0计算, A在GPU 1, B在GPU 0"
./nvlink_gemm -c 0 -a 1 -b 0 -m $M -n $N -k $K
echo ""
echo "======================================"
echo ""

# 测试场景3: 矩阵B在远程GPU
echo "【场景3】矩阵B在远程 - GPU 0计算, A在GPU 0, B在GPU 1"
./nvlink_gemm -c 0 -a 0 -b 1 -m $M -n $N -k $K
echo ""
echo "======================================"
echo ""

# 测试场景4: 两个矩阵都在远程GPU
echo "【场景4】两个矩阵都在远程 - GPU 0计算, A在GPU 1, B在GPU 2"
./nvlink_gemm -c 0 -a 1 -b 2 -m $M -n $N -k $K
echo ""
echo "======================================"
echo ""

# 测试场景5: 两个矩阵在同一个远程GPU
echo "【场景5】两个矩阵在同一远程GPU - GPU 0计算, A和B都在GPU 1"
./nvlink_gemm -c 0 -a 1 -b 1 -m $M -n $N -k $K
echo ""
echo "======================================"
echo ""

echo "所有测试完成！"

