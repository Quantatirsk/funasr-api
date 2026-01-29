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
EXPORT_TAR="false"
EXPORT_DIR="."
INTERACTIVE="true"
NO_CACHE="false"

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
    -e, --export          导出为 tar.gz 文件
    -o, --output DIR      导出目录 (默认: 当前目录)
    -r, --registry REG    镜像仓库 (默认: quantatrisk)
    -n, --no-cache        不使用缓存，强制重新安装依赖
    -y, --yes             跳过交互确认
    -h, --help            显示帮助

示例:
    ./build.sh                          # 交互式构建
    ./build.sh -t gpu -a amd64          # 构建 GPU 版本 (仅支持 amd64)
    ./build.sh -t cpu -a arm64          # 构建 CPU 版本 (arm64)
    ./build.sh -t cpu -a multi -p       # 构建 CPU 多架构并推送
    ./build.sh -t all -a amd64 -p       # 构建所有版本 (amd64) 并推送
    ./build.sh -t gpu -a amd64 -e       # 构建并导出为 tar.gz
    ./build.sh -t gpu -n                # 不使用缓存构建 GPU 版本

注意:
    GPU 版本仅支持 AMD64 架构 (Dockerfile.gpu.amd64)
    CPU 版本支持 AMD64 和 ARM64 多架构

    模型文件不包含在镜像中，需要通过 Volume 挂载:
    1. 运行 python scripts/download_models.py 下载模型
    2. 使用 docker-compose.yml 或添加 -v ./models:/root/.cache/modelscope
    详见: MODEL_SETUP.md

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

    echo -e "\n${BOLD}${prompt}${NC}" >&2
    for i in "${!options[@]}"; do
        echo -e "  ${GREEN}$((i+1))${NC}) ${options[$i]}" >&2
    done

    while true; do
        echo -ne "\n请输入选项 [1-${#options[@]}]: " >&2
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            echo $((choice - 1))
            return
        fi
        echo -e "${RED}无效选项，请重新输入${NC}" >&2
    done
}

# 交互式配置
interactive_config() {
    show_banner

    # 选择构建类型
    header "步骤 1/5: 选择构建类型"
    local types=("GPU 版本 (推荐生产环境)" "CPU 版本 (无 GPU 环境)" "全部构建 (GPU + CPU)")
    local type_idx=$(simple_select "选择要构建的镜像类型:" "${types[@]}")
    case $type_idx in
        0) BUILD_TYPE="gpu" ;;
        1) BUILD_TYPE="cpu" ;;
        2) BUILD_TYPE="all" ;;
    esac

    # 选择架构
    header "步骤 2/5: 选择目标架构"
    local archs=("amd64 (x86_64, 常见服务器/PC)" "arm64 (Apple Silicon, ARM 服务器)" "多架构 (amd64 + arm64)")
    local arch_idx=$(simple_select "选择目标架构:" "${archs[@]}")
    case $arch_idx in
        0) PLATFORM="linux/amd64" ;;
        1) PLATFORM="linux/arm64" ;;
        2) PLATFORM="linux/amd64,linux/arm64" ;;
    esac

    # 输入版本号
    header "步骤 3/5: 设置版本标签"
    echo -ne "请输入版本标签 [默认: latest]: "
    read -r input_version
    [ -n "$input_version" ] && VERSION="$input_version"

    # 是否推送
    header "步骤 4/5: 推送设置"
    local push_opts=("仅本地构建 (不推送)" "构建并推送到 Docker Hub")
    local push_idx=$(simple_select "选择推送选项:" "${push_opts[@]}")
    [ $push_idx -eq 1 ] && PUSH="true"

    # 是否导出为 tar.gz
    header "步骤 5/6: 导出设置"
    if [[ "$PLATFORM" == *","* ]]; then
        echo -e "${YELLOW}注意: 多架构构建不支持导出为 tar.gz${NC}"
        EXPORT_TAR="false"
    else
        local export_opts=("不导出" "导出为 tar.gz 文件")
        local export_idx=$(simple_select "是否导出镜像为 tar.gz:" "${export_opts[@]}")
        if [ $export_idx -eq 1 ]; then
            EXPORT_TAR="true"
            echo -ne "请输入导出目录 [默认: 当前目录]: "
            read -r input_dir
            [ -n "$input_dir" ] && EXPORT_DIR="$input_dir"
        fi
    fi

    # 是否使用缓存
    header "步骤 6/6: 构建缓存"
    local cache_opts=("使用缓存 (更快)" "不使用缓存 (强制重新安装依赖)")
    local cache_idx=$(simple_select "选择构建缓存策略:" "${cache_opts[@]}")
    [ $cache_idx -eq 1 ] && NO_CACHE="true"

    # 确认配置
    header "配置确认"
    echo -e "  构建类型:   ${CYAN}${BUILD_TYPE}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  镜像仓库:   ${CYAN}${REGISTRY}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"
    echo -e "  导出tar.gz: ${CYAN}$([ "$EXPORT_TAR" = "true" ] && echo "是 → ${EXPORT_DIR}" || echo "否")${NC}"
    echo -e "  使用缓存:   ${CYAN}$([ "$NO_CACHE" = "true" ] && echo "否 (强制重新安装)" || echo "是")${NC}"

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

    # CPU 版本使用不带 gpu 前缀的标签
    if [ "$VERSION" = "latest" ]; then
        tag="${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    info "构建 CPU 版本: $tag"
    info "目标架构: $PLATFORM"

    local build_args="--platform $PLATFORM -t $tag -f Dockerfile.cpu"

    # 添加 --no-cache 参数
    if [ "$NO_CACHE" = "true" ]; then
        build_args="$build_args --no-cache"
        info "已启用 --no-cache，将强制重新安装所有依赖"
    fi

    # 多架构只能 push
    if [[ "$PLATFORM" == *","* ]]; then
        if [ "$PUSH" != "true" ]; then
            warn "多架构构建需要 --push，将自动启用推送"
        fi
        build_args="$build_args --push"
    elif [ "$PUSH" = "true" ]; then
        build_args="$build_args --push"
    elif [ "$EXPORT_TAR" = "true" ]; then
        # 直接通过 buildx 导出为 tar
        local arch_suffix=""
        [ "$PLATFORM" = "linux/amd64" ] && arch_suffix="amd64"
        [ "$PLATFORM" = "linux/arm64" ] && arch_suffix="arm64"
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-cpu-${VERSION}-${arch_suffix}.tar"
        mkdir -p "$EXPORT_DIR"
        build_args="$build_args --output type=docker,dest=${tar_file}"
        info "将直接导出到: ${tar_file}.gz"
    else
        build_args="$build_args --load"
    fi

    docker buildx build $build_args .

    info "CPU 版本构建完成: $tag"

    # 如果使用了 --output 导出，压缩 tar 文件
    if [ "$EXPORT_TAR" = "true" ] && [[ "$PLATFORM" != *","* ]] && [ "$PUSH" != "true" ]; then
        local arch_suffix=""
        [ "$PLATFORM" = "linux/amd64" ] && arch_suffix="amd64"
        [ "$PLATFORM" = "linux/arm64" ] && arch_suffix="arm64"
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-cpu-${VERSION}-${arch_suffix}.tar"
        if [ -f "$tar_file" ]; then
            info "压缩: ${tar_file} → ${tar_file}.gz"
            gzip -f "$tar_file"
            local size=$(du -h "${tar_file}.gz" | cut -f1)
            info "导出成功: ${tar_file}.gz ($size)"
        fi
    fi
}

# 构建单架构 GPU 版本（使用统一标签，仅支持 AMD64）
build_gpu_single() {
    local tag="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"

    info "构建 GPU AMD64 版本: $tag"

    local build_args="--platform linux/amd64 -t $tag -f Dockerfile.gpu.amd64"

    # 添加 --no-cache 参数
    if [ "$NO_CACHE" = "true" ]; then
        build_args="$build_args --no-cache"
    fi

    if [ "$PUSH" = "true" ]; then
        build_args="$build_args --push"
    elif [ "$EXPORT_TAR" = "true" ]; then
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-gpu-${VERSION}-amd64.tar"
        mkdir -p "$EXPORT_DIR"
        build_args="$build_args --output type=docker,dest=${tar_file}"
    else
        build_args="$build_args --load"
    fi

    docker buildx build $build_args .
    info "GPU AMD64 版本构建完成: $tag"
}

# 构建 GPU 版本（统一入口，仅支持 AMD64）
build_gpu() {
    info "目标架构: $PLATFORM"

    # GPU 版本仅支持 AMD64
    if [[ "$PLATFORM" == *"arm64"* ]]; then
        error "GPU 版本不支持 ARM64 架构，请使用 CPU 版本或选择 AMD64 架构"
    fi

    # 仅支持单架构 AMD64
    if [ "$PLATFORM" != "linux/amd64" ]; then
        warn "GPU 版本仅支持 linux/amd64，将自动切换到 AMD64"
        PLATFORM="linux/amd64"
    fi

    build_gpu_single

    # 处理导出压缩
    if [ "$EXPORT_TAR" = "true" ] && [ "$PUSH" != "true" ]; then
        local tar_file="${EXPORT_DIR}/${IMAGE_NAME}-gpu-${VERSION}-amd64.tar"
        if [ -f "$tar_file" ]; then
            info "压缩: ${tar_file} → ${tar_file}.gz"
            gzip -f "$tar_file"
            local size=$(du -h "${tar_file}.gz" | cut -f1)
            info "导出成功: ${tar_file}.gz ($size)"
        fi
    fi
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
            -e|--export)
                EXPORT_TAR="true"
                shift
                ;;
            -o|--output)
                EXPORT_DIR="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -n|--no-cache)
                NO_CACHE="true"
                shift
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

    # 多架构不支持导出
    if [ "$EXPORT_TAR" = "true" ] && [[ "$PLATFORM" == *","* ]]; then
        warn "多架构构建不支持导出为 tar.gz，已禁用导出"
        EXPORT_TAR="false"
    fi

    # 检查 buildx
    check_buildx

    header "开始构建"
    echo -e "  构建类型:   ${CYAN}${BUILD_TYPE}${NC}"
    echo -e "  目标架构:   ${CYAN}${PLATFORM}${NC}"
    echo -e "  版本标签:   ${CYAN}${VERSION}${NC}"
    echo -e "  推送镜像:   ${CYAN}$([ "$PUSH" = "true" ] && echo "是" || echo "否")${NC}"
    echo -e "  导出tar.gz: ${CYAN}$([ "$EXPORT_TAR" = "true" ] && echo "是 → ${EXPORT_DIR}" || echo "否")${NC}"
    echo -e "  使用缓存:   ${CYAN}$([ "$NO_CACHE" = "true" ] && echo "否 (强制重新安装)" || echo "是")${NC}"

    # 显示 Dockerfile 信息
    if [[ "$BUILD_TYPE" == "gpu" || "$BUILD_TYPE" == "all" ]]; then
        echo -e "  GPU Dockerfile: ${CYAN}Dockerfile.gpu.amd64 (仅支持 AMD64)${NC}"
    fi
    if [[ "$BUILD_TYPE" == "cpu" || "$BUILD_TYPE" == "all" ]]; then
        echo -e "  CPU Dockerfile: ${CYAN}Dockerfile.cpu${NC}"
    fi
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

    # 显示构建的镜像（仅当加载到本地时）
    if [ "$PUSH" != "true" ] && [ "$EXPORT_TAR" != "true" ] && [[ "$PLATFORM" != *","* ]]; then
        info "已构建的镜像:"
        docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10
    fi

    # 显示导出的文件
    if [ "$EXPORT_TAR" = "true" ]; then
        echo ""
        info "导出的镜像文件:"
        ls -lh "${EXPORT_DIR}"/${IMAGE_NAME}*.tar.gz 2>/dev/null | tail -5
        echo ""
        info "使用以下命令加载镜像:"
        echo "  gunzip -c <file>.tar.gz | docker load"
    fi
}

main "$@"
