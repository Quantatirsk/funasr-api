#!/bin/bash
# FunASR-API Docker 镜像构建脚本（交互式）

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 默认配置
REGISTRY="quantatrisk"
IMAGE_NAME="funasr-api"
VERSION="latest"
BUILD_TYPE=""
PLATFORM=""
PUSH="false"
INTERACTIVE="true"

# 打印带颜色的消息
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
header() { echo -e "\n${BOLD}${BLUE}$1${NC}\n"; }

# 显示 banner
show_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
  ___            _   ___ ___     _   ___ ___
 | __|  _ _ _   /_\ / __| _ \   /_\ | _ \_ _|
 | _| || | ' \ / _ \\__ \   /  / _ \|  _/| |
 |_| \_,_|_||_/_/ \_\___/_|_\ /_/ \_\_| |___|

         Docker 镜像构建工具 v1.0
EOF
    echo -e "${NC}"
}

# 显示帮助
show_help() {
    cat << EOF
FunASR-API Docker 镜像构建脚本

用法: ./build.sh [选项]

选项:
    -t, --type TYPE       构建类型: cpu, gpu, all
    -a, --arch ARCH       架构: amd64, arm64, multi
    -v, --version VER     版本标签 (默认: latest)
    -p, --push            构建后推送到 Docker Hub
    -r, --registry REG    镜像仓库 (默认: quantatrisk)
    -y, --yes             跳过交互确认
    -h, --help            显示帮助

示例:
    ./build.sh                          # 交互式构建
    ./build.sh -t gpu -a amd64          # 构建 GPU 版本 (amd64)
    ./build.sh -t all -a multi -p       # 构建所有版本多架构并推送

EOF
}

# 选择菜单函数
select_option() {
    local prompt="$1"
    shift
    local options=("$@")
    local selected=0
    local key=""

    # 隐藏光标
    tput civis

    while true; do
        # 清除之前的输出
        echo -e "\n${BOLD}${prompt}${NC}"
        for i in "${!options[@]}"; do
            if [ $i -eq $selected ]; then
                echo -e "  ${GREEN}▶ ${options[$i]}${NC}"
            else
                echo -e "    ${options[$i]}"
            fi
        done

        # 读取按键
        read -rsn1 key
        case "$key" in
            A) # 上箭头
                ((selected--))
                [ $selected -lt 0 ] && selected=$((${#options[@]} - 1))
                ;;
            B) # 下箭头
                ((selected++))
                [ $selected -ge ${#options[@]} ] && selected=0
                ;;
            "") # Enter
                break
                ;;
        esac

        # 清除菜单行以便重绘
        for _ in "${options[@]}"; do
            tput cuu1
            tput el
        done
        tput cuu1
        tput el
    done

    # 显示光标
    tput cnorm

    echo $selected
}

# 简单选择（数字输入）
simple_select() {
    local prompt="$1"
    shift
    local options=("$@")

    echo -e "\n${BOLD}${prompt}${NC}"
    for i in "${!options[@]}"; do
        echo -e "  ${GREEN}$((i+1))${NC}) ${options[$i]}"
    done

    while true; do
        echo -ne "\n请输入选项 [1-${#options[@]}]: "
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo $((choice - 1))
            return
        fi
        echo -e "${RED}无效选项，请重新输入${NC}"
    done
}

# 交互式配置
interactive_config() {
    show_banner

    # 选择构建类型
    header "步骤 1/4: 选择构建类型"
    local types=("GPU 版本 (推荐生产环境)" "CPU 版本 (无 GPU 环境)" "全部构建 (GPU + CPU)")
    local type_idx=$(simple_select "选择要构建的镜像类型:" "${types[@]}")
    case $type_idx in
        0) BUILD_TYPE="gpu" ;;
        1) BUILD_TYPE="cpu" ;;
        2) BUILD_TYPE="all" ;;
    esac

    # 选择架构
    header "步骤 2/4: 选择目标架构"
    local archs=("amd64 (x86_64, 常见服务器/PC)" "arm64 (Apple Silicon, ARM 服务器)" "多架构 (amd64 + arm64)")
    local arch_idx=$(simple_select "选择目标架构:" "${archs[@]}")
    case $arch_idx in
        0) PLATFORM="linux/amd64" ;;
        1) PLATFORM="linux/arm64" ;;
        2) PLATFORM="linux/amd64,linux/arm64" ;;
    esac

    # 输入版本号
    header "步骤 3/4: 设置版本标签"
    echo -ne "请输入版本标签 [默认: latest]: "
    read -r input_version
    [ -n "$input_version" ] && VERSION="$input_version"

    # 是否推送
    header "步骤 4/4: 推送设置"
    local push_opts=("仅本地构建 (不推送)" "构建并推送到 Docker Hub")
    local push_idx=$(simple_select "选择推送选项:" "${push_opts[@]}")
    [ $push_idx -eq 1 ] && PUSH="true"

    # 确认配置
    header "配置确认"
    echo -e "  构建类型:   ${CYAN}${BUILD_TYPE}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  镜像仓库:   ${CYAN}${REGISTRY}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"

    echo ""
    echo -ne "确认开始构建? [Y/n]: "
    read -r confirm
    if [[ "$confirm" =~ ^[Nn] ]]; then
        echo -e "${YELLOW}已取消构建${NC}"
        exit 0
    fi
}

# 检查 buildx
check_buildx() {
    if ! docker buildx version &> /dev/null; then
        error "需要 Docker Buildx 支持多架构构建，请先安装"
    fi

    # 检查/创建 builder
    if ! docker buildx inspect funasr-builder &> /dev/null; then
        info "创建 buildx builder..."
        docker buildx create --name funasr-builder --use
    else
        docker buildx use funasr-builder
    fi
}

# 构建 CPU 版本
build_cpu() {
    local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    local tag_gpu="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"

    # CPU 版本使用不带 gpu 前缀的标签
    if [ "$VERSION" = "latest" ]; then
        tag="${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    info "构建 CPU 版本: $tag"
    info "目标架构: $PLATFORM"

    local build_args="--platform $PLATFORM -t $tag -f Dockerfile"

    if [ "$PUSH" = "true" ]; then
        build_args="$build_args --push"
    else
        build_args="$build_args --load"
    fi

    # 多架构只能 push，不能 load
    if [[ "$PLATFORM" == *","* ]] && [ "$PUSH" != "true" ]; then
        warn "多架构构建需要 --push，将自动启用推送"
        build_args="${build_args/--load/--push}"
    fi

    docker buildx build $build_args .

    info "CPU 版本构建完成: $tag"
}

# 构建 GPU 版本
build_gpu() {
    local tag="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"

    info "构建 GPU 版本: $tag"
    info "目标架构: $PLATFORM"

    local build_args="--platform $PLATFORM -t $tag -f Dockerfile.gpu"

    if [ "$PUSH" = "true" ]; then
        build_args="$build_args --push"
    else
        build_args="$build_args --load"
    fi

    # 多架构只能 push，不能 load
    if [[ "$PLATFORM" == *","* ]] && [ "$PUSH" != "true" ]; then
        warn "多架构构建需要 --push，将自动启用推送"
        build_args="${build_args/--load/--push}"
    fi

    docker buildx build $build_args .

    info "GPU 版本构建完成: $tag"
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                INTERACTIVE="false"
                shift 2
                ;;
            -a|--arch)
                case "$2" in
                    amd64) PLATFORM="linux/amd64" ;;
                    arm64) PLATFORM="linux/arm64" ;;
                    multi) PLATFORM="linux/amd64,linux/arm64" ;;
                    *) error "未知架构: $2 (可选: amd64, arm64, multi)" ;;
                esac
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--push)
                PUSH="true"
                shift
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -y|--yes)
                INTERACTIVE="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知选项: $1"
                ;;
        esac
    done
}

# 主流程
main() {
    parse_args "$@"

    # 如果没有通过参数指定，进入交互模式
    if [ -z "$BUILD_TYPE" ] && [ "$INTERACTIVE" = "true" ]; then
        interactive_config
    fi

    # 验证必要参数
    [ -z "$BUILD_TYPE" ] && error "请指定构建类型 (-t cpu|gpu|all)"
    [ -z "$PLATFORM" ] && PLATFORM="linux/amd64"  # 默认 amd64

    # 检查 buildx
    check_buildx

    header "开始构建"
    echo -e "  构建类型:   ${CYAN}${BUILD_TYPE}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"
    echo ""

    # 执行构建
    case $BUILD_TYPE in
        cpu)
            build_cpu
            ;;
        gpu)
            build_gpu
            ;;
        all)
            build_cpu
            build_gpu
            ;;
        *)
            error "未知构建类型: $BUILD_TYPE (可选: cpu, gpu, all)"
            ;;
    esac

    header "构建完成!"

    # 显示构建的镜像
    if [ "$PUSH" != "true" ] && [[ "$PLATFORM" != *","* ]]; then
        info "已构建的镜像:"
        docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10
    fi
}

main "$@"
