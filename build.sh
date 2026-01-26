#!/bin/bash
# FunASR-API Docker 镜像构建脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认配置
REGISTRY="quantatrisk"
IMAGE_NAME="funasr-api"
VERSION="${VERSION:-latest}"
PUSH="${PUSH:-false}"

# 打印带颜色的消息
info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 显示帮助
show_help() {
    cat << EOF
FunASR-API Docker 镜像构建脚本

用法: ./build.sh [选项]

选项:
    -t, --type TYPE     构建类型: cpu, gpu, all (默认: all)
    -v, --version VER   版本标签 (默认: latest)
    -p, --push          构建后推送到 Docker Hub
    -r, --registry REG  镜像仓库 (默认: quantatrisk)
    -h, --help          显示帮助

示例:
    ./build.sh                      # 构建 CPU 和 GPU 版本
    ./build.sh -t gpu               # 仅构建 GPU 版本
    ./build.sh -t gpu -v 1.0.0 -p   # 构建 GPU 版本并推送
    ./build.sh -t all -p            # 构建所有版本并推送

EOF
}

# 解析参数
BUILD_TYPE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
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
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "未知选项: $1"
            ;;
    esac
done

# 构建 CPU 版本
build_cpu() {
    info "构建 CPU 版本..."
    local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    local tag_latest="${REGISTRY}/${IMAGE_NAME}:latest"

    docker build -t "$tag" -f Dockerfile .

    if [ "$VERSION" != "latest" ]; then
        docker tag "$tag" "$tag_latest"
    fi

    info "CPU 版本构建完成: $tag"

    if [ "$PUSH" = "true" ]; then
        info "推送 CPU 版本..."
        docker push "$tag"
        [ "$VERSION" != "latest" ] && docker push "$tag_latest"
    fi
}

# 构建 GPU 版本
build_gpu() {
    info "构建 GPU 版本..."
    local tag="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
    local tag_latest="${REGISTRY}/${IMAGE_NAME}:gpu-latest"

    docker build -t "$tag" -f Dockerfile.gpu .

    if [ "$VERSION" != "latest" ]; then
        docker tag "$tag" "$tag_latest"
    fi

    info "GPU 版本构建完成: $tag"

    if [ "$PUSH" = "true" ]; then
        info "推送 GPU 版本..."
        docker push "$tag"
        [ "$VERSION" != "latest" ] && docker push "$tag_latest"
    fi
}

# 主流程
main() {
    info "=========================================="
    info "FunASR-API Docker 镜像构建"
    info "=========================================="
    info "构建类型: $BUILD_TYPE"
    info "版本标签: $VERSION"
    info "镜像仓库: $REGISTRY"
    info "推送镜像: $PUSH"
    info "=========================================="

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

    info "=========================================="
    info "构建完成!"
    info "=========================================="

    # 显示构建的镜像
    info "已构建的镜像:"
    docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10
}

main
