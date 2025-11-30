#!/bin/bash
# Cleanup script - removes build artifacts and old files
# Run this to clean up the project

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Cleanup Script ===${NC}"

# Clean build artifacts
echo -e "\n${BLUE}Cleaning build artifacts...${NC}"
rm -rf bin/*.o
rm -f bin/baseline bin/optimized_* bin/openmp bin/mpi
echo -e "${GREEN}✓ Build artifacts cleaned${NC}"

# Clean CUDA builds
if [ -d "src/cuda" ]; then
    echo -e "\n${BLUE}Cleaning CUDA builds...${NC}"
    cd src/cuda
    make clean 2>/dev/null || true
    cd ../..
    echo -e "${GREEN}✓ CUDA build cleaned${NC}"
fi

# Optional: Archive old directories
echo -e "\n${YELLOW}Optional cleanup (commented out by default):${NC}"
echo "  - src/algorithms/"
echo "  - src/cache_opt/"
echo "  - docs/"
echo "  - orca-website/"
echo ""
echo "To remove these, edit this script and uncomment the archive section"

# Uncomment to archive old directories
# if [ -d "old_archive" ]; then
#     rm -rf old_archive
# fi
# mkdir -p old_archive
# mv src/algorithms old_archive/ 2>/dev/null || true
# mv src/cache_opt old_archive/ 2>/dev/null || true
# mv docs old_archive/ 2>/dev/null || true
# mv orca-website old_archive/ 2>/dev/null || true
# echo -e "${GREEN}✓ Old directories archived to old_archive/${NC}"

echo -e "\n${GREEN}=== Cleanup complete ===${NC}"
