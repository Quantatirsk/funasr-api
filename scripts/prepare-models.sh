#!/bin/bash
#
# Model Export Tool for Offline Deployment
# Interactive version - KISS principle
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

OUTPUT_DIR="./models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print header
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}FunASR-API Model Export Tool${NC}                        ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     Export models for offline deployment                 ${CYAN}║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python
echo -e "${BLUE}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ python3 not found${NC}"
    exit 1
fi

# Check if running from project root or scripts directory
if [ -f "$PROJECT_ROOT/app/utils/download_models.py" ]; then
    cd "$PROJECT_ROOT"
elif [ -f "$SCRIPT_DIR/../app/utils/download_models.py" ]; then
    cd "$SCRIPT_DIR/.."
else
    echo -e "${RED}✗ Cannot find app/utils/download_models.py${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

echo -e "${GREEN}✓ Python OK${NC}"
echo ""

# Select models
echo -e "${BLUE}Select models to export:${NC}"
echo ""
echo -e "  ${CYAN}1)${NC} Auto (recommended)"
echo -e "     Detect GPU VRAM and export appropriate models"
echo ""
echo -e "  ${CYAN}2)${NC} All models"
echo -e "     Qwen3-ASR 1.7B + 0.6B + Paraformer + VAD + CAM++"
echo ""
echo -e "  ${CYAN}3)${NC} Paraformer only"
echo -e "     Lightweight, CPU/GPU compatible (~3GB)"
echo ""
echo -e "  ${CYAN}4)${NC} Qwen3-ASR 0.6B + Paraformer"
echo -e "     Small VRAM option (~6GB)"
echo ""
echo -e "  ${CYAN}5)${NC} Qwen3-ASR 1.7B + Paraformer"
echo -e "     Best quality, requires 16GB+ VRAM (~10GB)"
echo ""

read -p "Enter choice [1-5]: " choice
case $choice in
    1) ENABLED_MODELS="auto"; MODEL_DESC="Auto-detected" ;;
    2) ENABLED_MODELS="all"; MODEL_DESC="All models" ;;
    3) ENABLED_MODELS="paraformer-large"; MODEL_DESC="Paraformer only" ;;
    4) ENABLED_MODELS="qwen3-asr-0.6b,paraformer-large"; MODEL_DESC="Qwen3 0.6B + Paraformer" ;;
    5) ENABLED_MODELS="qwen3-asr-1.7b,paraformer-large"; MODEL_DESC="Qwen3 1.7B + Paraformer" ;;
    *) echo -e "${RED}Invalid choice, using auto${NC}"; ENABLED_MODELS="auto"; MODEL_DESC="Auto-detected" ;;
esac

echo ""
echo -e "${GREEN}Selected: ${MODEL_DESC}${NC}"
echo ""

# Confirm
echo -e "${YELLOW}Export settings:${NC}"
echo -e "  Models: ${CYAN}${MODEL_DESC}${NC}"
echo -e "  Output: ${CYAN}${OUTPUT_DIR}/${NC}"
echo ""
read -p "Start export? [Y/n]: " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""

# Export
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Exporting models...${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Remove existing models dir to ensure clean state
rm -rf "${OUTPUT_DIR}"

# Run Python export
export ENABLED_MODELS="${ENABLED_MODELS}"
python3 -c "
import sys
sys.path.insert(0, 'app/utils')
from download_models import download_models
success = download_models(auto_mode=False, export_dir='${OUTPUT_DIR}')
sys.exit(0 if success else 1)
"

echo ""

# Package
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Packaging...${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

PACKAGE="funasr-models-$(date +%Y%m%d-%H%M).tar.gz"

# Use pigz for multi-threaded compression if available
if command -v pigz &> /dev/null; then
    echo "Using pigz for multi-threaded compression..."
    # Get CPU cores (cross-platform: Linux and macOS)
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif command -v sysctl &> /dev/null; then
        CPU_CORES=$(sysctl -n hw.ncpu)
    else
        CPU_CORES=4
    fi
    tar -cf - "${OUTPUT_DIR}" | pigz -p "${CPU_CORES}" > "${PACKAGE}"
else
    echo "pigz not found, using standard gzip..."
    tar -czf "${PACKAGE}" "${OUTPUT_DIR}"
fi

SIZE=$(du -sh "${PACKAGE}" | cut -f1)

# Done
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Export Complete!                                          ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Package: ${CYAN}${PACKAGE}${NC}"
echo -e "Size:    ${CYAN}${SIZE}${NC}"
echo ""
echo -e "${YELLOW}To deploy on an offline server:${NC}"
echo ""
echo -e "1. Copy package:  ${CYAN}scp ${PACKAGE} user@server:/opt/funasr-api/${NC}"
echo -e "2. Extract:       ${CYAN}tar -xzvf ${PACKAGE}${NC}"
echo -e "3. Start service: ${CYAN}docker-compose up -d${NC}"
echo ""
