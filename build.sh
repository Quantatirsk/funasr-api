#!/bin/bash
#
# FunASR-API Docker Build Tool
# Supports both interactive (menu) and parameter-driven (CI) modes.

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

REGISTRY="quantatrisk"
IMAGE_NAME="funasr-api"
VERSION="latest"
BUILD_TYPE="all"
PLATFORM="linux/amd64"
PUSH="false"
EXPORT_TAR="false"
EXPORT_DIR="."
NO_CACHE="false"
LANG_MODE="auto"

# =============================================================================
# Localization
# =============================================================================

_resolve_lang() {
  if [[ "$LANG_MODE" != "auto" ]]; then
    printf "%s" "$LANG_MODE"
    return
  fi
  local sys_lang="${LANG:-${LC_ALL:-}}"
  if [[ "$sys_lang" == zh* || "$sys_lang" == Zh* || "$sys_lang" == ZH* ]]; then
    printf "zh"
  else
    printf "en"
  fi
}

_msg() {
  local key="$1"
  local lang
  lang="$(_resolve_lang)"
  case "$lang" in
    zh)
      case "$key" in
        title)              echo "FunASR-API Docker 构建工具" ;;
        subtitle)           echo "支持交互式菜单与命令行参数两种模式" ;;
        opt_build_type)     echo "构建目标" ;;
        opt_arch)           echo "目标架构" ;;
        opt_version)        echo "镜像版本" ;;
        opt_push)           echo "推送镜像" ;;
        opt_export)         echo "导出 tar.gz" ;;
        opt_output)         echo "输出目录" ;;
        opt_registry)       echo "仓库命名空间" ;;
        opt_no_cache)       echo "禁用缓存" ;;
        opt_lang)           echo "语言" ;;
        current_value)      echo "当前值" ;;
        menu_prompt)        echo "输入编号修改选项，直接回车开始构建" ;;
        enter_build_type)   echo "选择构建目标 (cpu / gpu / all)" ;;
        enter_arch)         echo "选择架构 (amd64 / arm64 / multi)" ;;
        enter_version)      echo "输入版本号" ;;
        enter_push)         echo "是否推送镜像到仓库 (y/n)" ;;
        enter_export)       echo "是否导出为 tar.gz (y/n)" ;;
        enter_output)       echo "输入导出目录" ;;
        enter_registry)     echo "输入仓库命名空间" ;;
        enter_no_cache)     echo "是否禁用构建缓存 (y/n)" ;;
        enter_lang)         echo "选择语言 (zh / en / auto)" ;;
        info_start)         echo "开始构建..." ;;
        info_build_type)    echo "构建目标" ;;
        info_platform)      echo "平台" ;;
        info_version)       echo "版本" ;;
        info_push)          echo "推送" ;;
        info_export)        echo "导出" ;;
        info_no_cache)      echo "禁用缓存" ;;
        info_done)          echo "构建完成" ;;
        info_recent_images) echo "最近构建的镜像" ;;
        info_creating_builder) echo "创建 buildx builder" ;;
        help_desc)          echo "FunASR-API Docker 构建工具" ;;
        help_usage)         echo "用法" ;;
        help_options)       echo "选项" ;;
        help_examples)      echo "示例" ;;
        err_invalid)        echo "无效输入" ;;
        err_unknown_opt)    echo "未知选项" ;;
        err_gpu_arm64)      echo "GPU 构建仅支持 amd64 架构" ;;
        err_buildx)         echo "需要 Docker Buildx" ;;
        val_yes)            echo "是" ;;
        val_no)             echo "否" ;;
        val_auto)           echo "自动" ;;
      esac
      ;;
    *)
      case "$key" in
        title)              echo "FunASR-API Docker Build Tool" ;;
        subtitle)           echo "Supports both interactive menu and CLI parameter modes" ;;
        opt_build_type)     echo "Build Target" ;;
        opt_arch)           echo "Architecture" ;;
        opt_version)        echo "Version" ;;
        opt_push)           echo "Push Image" ;;
        opt_export)         echo "Export tar.gz" ;;
        opt_output)         echo "Output Dir" ;;
        opt_registry)       echo "Registry" ;;
        opt_no_cache)       echo "No Cache" ;;
        opt_lang)           echo "Language" ;;
        current_value)      echo "Current" ;;
        menu_prompt)        echo "Enter number to edit, or press Enter to build" ;;
        enter_build_type)   echo "Select build target (cpu / gpu / all)" ;;
        enter_arch)         echo "Select architecture (amd64 / arm64 / multi)" ;;
        enter_version)      echo "Enter version tag" ;;
        enter_push)         echo "Push to registry? (y/n)" ;;
        enter_export)       echo "Export as tar.gz? (y/n)" ;;
        enter_output)       echo "Enter output directory" ;;
        enter_registry)     echo "Enter registry namespace" ;;
        enter_no_cache)     echo "Disable build cache? (y/n)" ;;
        enter_lang)         echo "Select language (zh / en / auto)" ;;
        info_start)         echo "Starting build..." ;;
        info_build_type)    echo "Build type" ;;
        info_platform)      echo "Platform" ;;
        info_version)       echo "Version" ;;
        info_push)          echo "Push" ;;
        info_export)        echo "Export" ;;
        info_no_cache)      echo "No cache" ;;
        info_done)          echo "Build complete" ;;
        info_recent_images) echo "Recent images" ;;
        info_creating_builder) echo "Creating buildx builder" ;;
        help_desc)          echo "FunASR-API Docker build wrapper" ;;
        help_usage)         echo "Usage" ;;
        help_options)       echo "Options" ;;
        help_examples)      echo "Examples" ;;
        err_invalid)        echo "Invalid input" ;;
        err_unknown_opt)    echo "Unknown option" ;;
        err_gpu_arm64)      echo "GPU build only supports amd64" ;;
        err_buildx)         echo "Docker Buildx is required" ;;
        val_yes)            echo "yes" ;;
        val_no)             echo "no" ;;
        val_auto)           echo "auto" ;;
      esac
      ;;
  esac
}

_yes_no_label() {
  if [[ "$1" == "true" ]]; then
    _msg val_yes
  else
    _msg val_no
  fi
}

# =============================================================================
# Helpers
# =============================================================================

info() { echo "[INFO] $1"; }
warn() { echo "[WARN] $1"; }
die() { echo "[ERROR] $1" >&2; exit 1; }

parse_arch() {
  case "$1" in
    amd64) echo "linux/amd64" ;;
    arm64) echo "linux/arm64" ;;
    multi) echo "linux/amd64,linux/arm64" ;;
    *) die "$(_msg err_invalid): $1" ;;
  esac
}

arch_label() {
  case "$1" in
    linux/amd64) echo "amd64" ;;
    linux/arm64) echo "arm64" ;;
    linux/amd64,linux/arm64) echo "multi" ;;
    *) echo "$1" ;;
  esac
}

ensure_buildx() {
  docker buildx version >/dev/null 2>&1 || die "$(_msg err_buildx)"
  if ! docker buildx inspect funasr-builder >/dev/null 2>&1; then
    info "$(_msg info_creating_builder): funasr-builder"
    docker buildx create --name funasr-builder --driver docker-container --use >/dev/null
  else
    docker buildx use funasr-builder >/dev/null
  fi
}

export_compressor() {
  if command -v pigz >/dev/null 2>&1; then
    echo "pigz -f"
  else
    echo "gzip -f"
  fi
}

# =============================================================================
# Build Core
# =============================================================================

build_image() {
  local target="$1"
  local dockerfile="$2"
  local image_tag="$3"
  local platform="$4"

  local args=(buildx build --platform "$platform" -f "$dockerfile" -t "$image_tag")

  if [[ "$NO_CACHE" == "true" ]]; then
    args+=(--no-cache)
  fi

  if [[ "$platform" == *","* ]]; then
    if [[ "$EXPORT_TAR" == "true" ]]; then
      warn "Multi-arch build cannot export tar.gz, skipping export"
    fi
    args+=(--push)
  elif [[ "$PUSH" == "true" ]]; then
    args+=(--push)
  elif [[ "$EXPORT_TAR" == "true" ]]; then
    mkdir -p "$EXPORT_DIR"
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar_path="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    args+=(--output "type=docker,dest=${tar_path}")
  else
    args+=(--load)
  fi

  info "Building ${target}: ${image_tag} (${platform})"
  docker "${args[@]}" .

  if [[ "$EXPORT_TAR" == "true" && "$platform" != *","* && "$PUSH" != "true" ]]; then
    local suffix="${target}-${VERSION}-$(basename "$platform")"
    local tar_path="${EXPORT_DIR}/${IMAGE_NAME}-${suffix}.tar"
    if [[ -f "$tar_path" ]]; then
      info "Compressing ${tar_path}"
      $(export_compressor) "$tar_path"
      info "Exported ${tar_path}.gz"
    fi
  fi
}

build_cpu() {
  local tag="${REGISTRY}/${IMAGE_NAME}:${VERSION}"
  if [[ "$VERSION" == "latest" ]]; then
    tag="${REGISTRY}/${IMAGE_NAME}:cpu-latest"
  fi
  build_image "cpu" "Dockerfile.cpu" "$tag" "$PLATFORM"
}

build_gpu() {
  if [[ "$PLATFORM" == *"arm64"* ]]; then
    die "$(_msg err_gpu_arm64)"
  fi
  local platform="linux/amd64"
  local tag="${REGISTRY}/${IMAGE_NAME}:gpu-${VERSION}"
  build_image "gpu" "Dockerfile.gpu" "$tag" "$platform"
}

# =============================================================================
# Interactive Mode
# =============================================================================

_prompt() {
  local val
  read -r -p "$1" val
  printf "%s" "$val"
}

_show_menu() {
  local yn_push yn_export yn_cache
  yn_push="$(_yes_no_label "$PUSH")"
  yn_export="$(_yes_no_label "$EXPORT_TAR")"
  yn_cache="$(_yes_no_label "$NO_CACHE")"
  local arch
  arch="$(arch_label "$PLATFORM")"
  local lang_disp
  case "$LANG_MODE" in
    zh) lang_disp="zh" ;;
    en) lang_disp="en" ;;
    *)  lang_disp="$(_msg val_auto) ($(_resolve_lang))" ;;
  esac

cat <<EOF

========================================
  $(_msg title)
========================================
  $(_msg subtitle)

  1. $(_msg opt_build_type)   : ${BUILD_TYPE}
  2. $(_msg opt_arch)         : ${arch}
  3. $(_msg opt_version)      : ${VERSION}
  4. $(_msg opt_push)         : ${yn_push}
  5. $(_msg opt_export)       : ${yn_export}
  6. $(_msg opt_output)       : ${EXPORT_DIR}
  7. $(_msg opt_registry)     : ${REGISTRY}
  8. $(_msg opt_no_cache)     : ${yn_cache}
  9. $(_msg opt_lang)         : ${lang_disp}

----------------------------------------
$(_msg menu_prompt)
EOF
}

_read_choice() {
  local choice
  read -r -p "> " choice
  printf "%s" "$choice"
}

_edit_build_type() {
  local val
  val="$(_prompt "$(_msg enter_build_type) [${BUILD_TYPE}]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    cpu|gpu|all) BUILD_TYPE="$val" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

_edit_arch() {
  local val
  val="$(_prompt "$(_msg enter_arch) [$(arch_label "$PLATFORM")]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    amd64|arm64|multi) PLATFORM="$(parse_arch "$val")" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

_edit_version() {
  local val
  val="$(_prompt "$(_msg enter_version) [${VERSION}]: ")"
  [[ -n "$val" ]] && VERSION="$val"
}

_edit_push() {
  local val
  val="$(_prompt "$(_msg enter_push) [$(_yes_no_label "$PUSH")]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    [Yy]|[Yy][Ee][Ss]|是) PUSH="true" ;;
    [Nn]|[Nn][Oo]|否) PUSH="false" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

_edit_export() {
  local val
  val="$(_prompt "$(_msg enter_export) [$(_yes_no_label "$EXPORT_TAR")]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    [Yy]|[Yy][Ee][Ss]|是) EXPORT_TAR="true" ;;
    [Nn]|[Nn][Oo]|否) EXPORT_TAR="false" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

_edit_output() {
  local val
  val="$(_prompt "$(_msg enter_output) [${EXPORT_DIR}]: ")"
  [[ -n "$val" ]] && EXPORT_DIR="$val"
}

_edit_registry() {
  local val
  val="$(_prompt "$(_msg enter_registry) [${REGISTRY}]: ")"
  [[ -n "$val" ]] && REGISTRY="$val"
}

_edit_no_cache() {
  local val
  val="$(_prompt "$(_msg enter_no_cache) [$(_yes_no_label "$NO_CACHE")]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    [Yy]|[Yy][Ee][Ss]|是) NO_CACHE="true" ;;
    [Nn]|[Nn][Oo]|否) NO_CACHE="false" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

_edit_lang() {
  local val
  val="$(_prompt "$(_msg enter_lang) [${LANG_MODE}]: ")"
  [[ -z "$val" ]] && return
  case "$val" in
    zh|en|auto) LANG_MODE="$val" ;;
    *) warn "$(_msg err_invalid): ${val}" ;;
  esac
}

interactive_mode() {
  while true; do
    _show_menu
    local choice
    choice="$(_read_choice)"
    case "$choice" in
      1) _edit_build_type ;;
      2) _edit_arch ;;
      3) _edit_version ;;
      4) _edit_push ;;
      5) _edit_export ;;
      6) _edit_output ;;
      7) _edit_registry ;;
      8) _edit_no_cache ;;
      9) _edit_lang ;;
      "") break ;;
      *) warn "$(_msg err_invalid): ${choice}" ;;
    esac
  done

  info "$(_msg info_start)"
  info "  $(_msg info_build_type): ${BUILD_TYPE}"
  info "  $(_msg info_platform): ${PLATFORM}"
  info "  $(_msg info_version): ${VERSION}"
  info "  $(_msg info_push): $(_yes_no_label "$PUSH")"
  info "  $(_msg info_export): $(_yes_no_label "$EXPORT_TAR")"
  info "  $(_msg info_no_cache): $(_yes_no_label "$NO_CACHE")"
}

# =============================================================================
# CLI Mode
# =============================================================================

show_help() {
cat <<EOF
$(_msg help_desc)

$(_msg help_usage):
  ./build.sh [options]

$(_msg help_options):
  -t, --type TYPE       $(_msg opt_build_type): cpu, gpu, all (default: all)
  -a, --arch ARCH       $(_msg opt_arch): amd64, arm64, multi (default: amd64)
  -v, --version VER     $(_msg opt_version) (default: latest)
  -p, --push            $(_msg opt_push)
  -e, --export          $(_msg opt_export)
  -o, --output DIR      $(_msg opt_output) (default: .)
  -r, --registry REG    $(_msg opt_registry) (default: quantatrisk)
  -n, --no-cache        $(_msg opt_no_cache)
  -l, --lang LANG       $(_msg opt_lang): zh, en, auto (default: auto)
  -h, --help            Show this help

$(_msg help_examples):
  ./build.sh
  ./build.sh -t gpu -a amd64
  ./build.sh -t cpu -a multi -p
  ./build.sh -t all -v 1.0.0 -p -l zh
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--type)
        BUILD_TYPE="$2"
        shift 2
        ;;
      -a|--arch)
        PLATFORM="$(parse_arch "$2")"
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
      -l|--lang)
        LANG_MODE="$2"
        shift 2
        ;;
      -h|--help)
        show_help
        exit 0
        ;;
      *)
        die "$(_msg err_unknown_opt): $1"
        ;;
    esac
  done
}

validate_args() {
  case "$BUILD_TYPE" in
    cpu|gpu|all) ;;
    *) die "$(_msg err_invalid): $(_msg opt_build_type) = ${BUILD_TYPE}" ;;
  esac
}

# =============================================================================
# Main
# =============================================================================

main() {
  if [[ $# -eq 0 && -t 0 ]]; then
    interactive_mode
  else
    parse_args "$@"
    validate_args
  fi

  ensure_buildx

  case "$BUILD_TYPE" in
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
  esac

  if [[ "$PUSH" != "true" && "$EXPORT_TAR" != "true" ]]; then
    info "$(_msg info_recent_images):"
    docker images | grep "${REGISTRY}/${IMAGE_NAME}" | head -10 || true
  fi

  info "$(_msg info_done)"
}

main "$@"
