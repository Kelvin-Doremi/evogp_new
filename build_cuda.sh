#!/bin/bash
# evogp CUDA 扩展编译脚本
# 在项目根目录执行此脚本以编译 evogp_cuda 模块

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> 正在编译 evogp CUDA 扩展..."
echo "    项目目录: $SCRIPT_DIR"
echo ""

# 检查 nvcc 是否可用
if ! command -v nvcc > /dev/null 2>&1; then
    echo "错误: 未找到 nvcc，请确保 CUDA Toolkit 已安装并加入 PATH"
    exit 1
fi

echo "nvcc 版本: $(nvcc --version | grep release)"
echo ""

# RTX 4060 Ti 为 Ada 架构，计算能力 8.9，仅编译此架构可加速
export TORCH_CUDA_ARCH_LIST="8.9"

# 仅编译 CUDA 扩展，不安装（产物输出到 evogp/ 目录）
python setup.py build_ext --inplace

echo ""
echo "==> 编译完成！"
