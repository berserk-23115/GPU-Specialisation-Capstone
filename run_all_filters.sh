#!/bin/bash

# Script to run the video processor with all available filters

# Exit on error
set -e

# Check if input video is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_video_file>"
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="./data/output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of filters to process
FILTERS=(
    "blur"
    "sharpen"
    "edge_detect"
    "emboss"
    "sepia"
    "grayscale"
    "negative"
    "thermal"
    "night_vision"
)

# Process video with each filter
echo "Processing video with all filters..."

for filter in "${FILTERS[@]}"; do
    echo "Applying filter: $filter"
    output_file="$OUTPUT_DIR/${filter}_output.mp4"
    
    # Run the processor with current filter
    ./build/video_processor --input "$INPUT_VIDEO" --output "$output_file" --filter "$filter" --intensity 0.8
    
    echo "Output saved to: $output_file"
    echo "------------------------"
done

# Run benchmark
echo "Running performance benchmark..."
./build/video_processor --input "$INPUT_VIDEO" --benchmark

echo "All processing completed successfully!" 