# Data Directory

This directory contains input and output files for the CUDA Video Processor.

## Structure

- `input/`: Place your input video files in this directory
- `output/`: Processed videos will be saved in this directory

## Supported Input Formats

The application supports various video formats through OpenCV:

- `.mp4`: MPEG-4 video
- `.avi`: AVI video
- `.mov`: QuickTime video
- `.mkv`: Matroska video
- And many other formats supported by OpenCV

## Sample Data

To get started, you can place a sample video in the `input/` directory. You can use:

1. Your own video files
2. Camera input (by specifying camera index instead of file)
3. Sample videos from online repositories

## Output Files

When processing videos, the output files will be saved in the `output/` directory with a naming convention based on the filter applied:

```
<filter_name>_output.mp4
```

For example: `blur_output.mp4`, `thermal_output.mp4`, etc. 