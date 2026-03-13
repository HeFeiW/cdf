#!/bin/bash
# CDF可视化辅助脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CDF接触构型可视化${NC}"
echo "======================================"

# 检查参数
if [ "$1" == "generate" ]; then
    echo -e "${YELLOW}模式: 生成接触构型${NC}"
    
    # 生成构型
    python3 visualize_cdf.py --mode generate \
        --robot leaphand \
        --serial_idx 0 \
        --obj_mesh /workspace/cdf/models/004/textured.obj \
        --obj_position 0.0 0.0 0.4 \
        --obj_orientation 0.0 0.0 0.0 \
        --obj_scale 1.0 \
        --num_samples 1000 \
        --num_configs 50 \
        --num_iterations 10 \
        --device cuda \
        --output contact_configs.json
    
    echo -e "${GREEN}构型生成完成！${NC}"
    echo "下一步运行: ./visualize_cdf.sh visualize"

elif [ "$1" == "visualize" ]; then
    echo -e "${YELLOW}模式: 可视化${NC}"
    
    # 检查是否已生成构型文件
    if [ ! -f "contact_configs.json" ]; then
        echo -e "${RED}错误: 找不到 contact_configs.json${NC}"
        echo "请先运行: ./visualize_cdf.sh generate"
        exit 1
    fi
    
    # 启动RViz和可视化节点
    echo "启动RViz..."
    
    # 在后台启动RViz
    cd ../../RDF_ori/show_urdf_ws
    source install/setup.bash
    
    # 启动RViz（后台）
    rviz2 &
    RVIZ_PID=$!
    
    # 等待RViz启动
    sleep 3
    
    # 启动可视化节点
    cd ../../cdf/frankaemika
    echo -e "${GREEN}启动可视化节点...${NC}"
    echo "使用键盘控制: n(下一个), p(上一个), q(退出)"
    
    python3 visualize_cdf.py --mode visualize --config_file contact_configs.json
    
    # 清理
    echo "关闭RViz..."
    kill $RVIZ_PID 2>/dev/null
    
else
    echo "用法:"
    echo "  $0 generate   - 生成接触构型"
    echo "  $0 visualize  - 可视化构型"
    echo ""
    echo "示例:"
    echo "  $0 generate"
    echo "  $0 visualize"
fi

echo "======================================"
