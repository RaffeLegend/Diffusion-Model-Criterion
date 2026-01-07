#!/bin/bash
#===============================================================================
# Diffusion Model Evaluation Runner
# 
# Usage:
#   ./run.sh                    # 显示帮助
#   ./run.sh pde ./images       # 使用 PDE 方法评估 ./images 目录
#   ./run.sh clip ./images      # 使用 CLIP 方法评估
#===============================================================================

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
# 根据你的环境修改这些默认值

DEFAULT_DEVICE="cuda"
DEFAULT_BATCH_SIZE=4
DEFAULT_NUM_NOISE=8
DEFAULT_TIME_FRAC=0.01
DEFAULT_IMAGE_SIZE=512
DEFAULT_OUTPUT_DIR="results"

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==================== 帮助信息 ====================
show_help() {
    echo "==============================================================================="
    echo "  Diffusion Model Evaluation Framework"
    echo "==============================================================================="
    echo ""
    echo "用法:"
    echo "  $0 <criterion> <image_dir> [options]"
    echo ""
    echo "参数:"
    echo "  criterion     评估方法: pde, clip, 或其他已注册的方法"
    echo "  image_dir     包含图片的目录路径"
    echo ""
    echo "选项:"
    echo "  -b, --batch-size SIZE    批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  -n, --num-noise NUM      每张图片的噪声样本数 (默认: $DEFAULT_NUM_NOISE)"
    echo "  -t, --time-frac FRAC     扩散时间步比例 (默认: $DEFAULT_TIME_FRAC)"
    echo "  -s, --image-size SIZE    图片尺寸 (默认: $DEFAULT_IMAGE_SIZE)"
    echo "  -o, --output-dir DIR     输出目录 (默认: $DEFAULT_OUTPUT_DIR)"
    echo "  -d, --device DEVICE      设备: cuda 或 cpu (默认: $DEFAULT_DEVICE)"
    echo "  -r, --recursive          递归搜索子目录"
    echo "  --return-terms           返回详细的中间指标"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本用法"
    echo "  $0 pde ./test_images"
    echo ""
    echo "  # 使用 CLIP 方法，自定义参数"
    echo "  $0 clip ./test_images -b 8 -n 16 -o results/clip_exp1"
    echo ""
    echo "  # 递归搜索子目录"
    echo "  $0 pde ./dataset -r"
    echo ""
    echo "  # 查看可用的评估方法"
    echo "  $0 list"
    echo ""
    echo "==============================================================================="
}

# ==================== 列出可用方法 ====================
list_criteria() {
    print_info "可用的评估方法:"
    python main.py --list-criteria
}

# ==================== 检查环境 ====================
check_environment() {
    # 检查 Python
    if ! command -v python &> /dev/null; then
        print_error "Python 未安装或不在 PATH 中"
        exit 1
    fi
    
    # 检查 CUDA (如果使用 GPU)
    if [ "$DEVICE" = "cuda" ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            print_warning "CUDA 不可用，将使用 CPU"
            DEVICE="cpu"
        fi
    fi
    
    # 检查必要的包
    python -c "import torch, diffusers, transformers" 2>/dev/null || {
        print_error "缺少必要的 Python 包。请运行:"
        echo "  pip install torch diffusers transformers imageio pillow tqdm pandas"
        exit 1
    }
}

# ==================== 主执行函数 ====================
run_evaluation() {
    local CRITERION=$1
    local IMAGE_DIR=$2
    shift 2
    
    # 默认值
    BATCH_SIZE=$DEFAULT_BATCH_SIZE
    NUM_NOISE=$DEFAULT_NUM_NOISE
    TIME_FRAC=$DEFAULT_TIME_FRAC
    IMAGE_SIZE=$DEFAULT_IMAGE_SIZE
    OUTPUT_DIR=$DEFAULT_OUTPUT_DIR
    DEVICE=$DEFAULT_DEVICE
    RECURSIVE=""
    RETURN_TERMS=""
    
    # 解析选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -n|--num-noise)
                NUM_NOISE="$2"
                shift 2
                ;;
            -t|--time-frac)
                TIME_FRAC="$2"
                shift 2
                ;;
            -s|--image-size)
                IMAGE_SIZE="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -d|--device)
                DEVICE="$2"
                shift 2
                ;;
            -r|--recursive)
                RECURSIVE="--recursive"
                shift
                ;;
            --return-terms)
                RETURN_TERMS="--return-terms"
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查目录是否存在
    if [ ! -d "$IMAGE_DIR" ]; then
        print_error "目录不存在: $IMAGE_DIR"
        exit 1
    fi
    
    # 检查环境
    check_environment
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 显示配置
    echo ""
    echo "==============================================================================="
    print_info "评估配置:"
    echo "  方法:         $CRITERION"
    echo "  图片目录:     $IMAGE_DIR"
    echo "  输出目录:     $OUTPUT_DIR"
    echo "  设备:         $DEVICE"
    echo "  批处理大小:   $BATCH_SIZE"
    echo "  噪声样本数:   $NUM_NOISE"
    echo "  时间步比例:   $TIME_FRAC"
    echo "  图片尺寸:     $IMAGE_SIZE"
    echo "==============================================================================="
    echo ""
    
    # 运行评估
    print_info "开始评估..."
    
    python main.py \
        --dir "$IMAGE_DIR" \
        --criterion "$CRITERION" \
        --batch-size "$BATCH_SIZE" \
        --num-noise "$NUM_NOISE" \
        --time-frac "$TIME_FRAC" \
        --image-size "$IMAGE_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        $RECURSIVE \
        $RETURN_TERMS
    
    if [ $? -eq 0 ]; then
        print_success "评估完成！结果保存在: $OUTPUT_DIR"
    else
        print_error "评估失败"
        exit 1
    fi
}

# ==================== 批量实验脚本 ====================
run_batch_experiments() {
    # 示例：批量运行多个实验
    local IMAGE_DIR=$1
    local BASE_OUTPUT="results/batch_$(date +%Y%m%d_%H%M%S)"
    
    print_info "运行批量实验..."
    
    # 实验 1: PDE 方法
    run_evaluation pde "$IMAGE_DIR" -o "${BASE_OUTPUT}/pde" --return-terms
    
    # 实验 2: CLIP 方法  
    run_evaluation clip "$IMAGE_DIR" -o "${BASE_OUTPUT}/clip" --return-terms
    
    print_success "批量实验完成！结果保存在: $BASE_OUTPUT"
}

# ==================== 主入口 ====================
main() {
    # 切换到脚本所在目录
    cd "$(dirname "$0")"
    
    # 无参数显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 处理特殊命令
    case $1 in
        -h|--help|help)
            show_help
            exit 0
            ;;
        list)
            list_criteria
            exit 0
            ;;
        batch)
            if [ -z "$2" ]; then
                print_error "请指定图片目录"
                exit 1
            fi
            run_batch_experiments "$2"
            exit 0
            ;;
        *)
            # 检查是否提供了足够的参数
            if [ $# -lt 2 ]; then
                print_error "缺少参数"
                show_help
                exit 1
            fi
            run_evaluation "$@"
            ;;
    esac
}

# 执行主函数
main "$@"

# # 基本用法
# ./run.sh pde ./images                    # PDE 方法
# ./run.sh clip ./images                   # CLIP 方法

# # 自定义参数
# ./run.sh pde ./images -b 8 -n 16 -o results/exp1

# # 查看可用方法
# ./run.sh list

# # 批量运行所有方法
# ./run.sh batch ./images