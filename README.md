# Video Scene Splitter

A Python tool that automatically detects and splits videos into separate scenes using OpenAI's GPT-4 Vision API and OpenCV.

## Features
- Combines GPT-4 Vision and OpenCV for accurate scene detection
- Splits video into separate files for each scene
- Generates detailed scene analysis report
- Supports MP4 video format

## ❤️ Support & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! **[Support me on Patreon](https://www.patreon.com/c/echohive42/membership)** to get:

- 🎯 Access to 400+ AI projects (and growing weekly!)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools

## 🔧 Prerequisites

## Requirements
- Python 3.6+
- OpenAI API key
- Input video file (video.mp4)

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
Export your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage
1. Place your video file as `video.mp4` in the project directory
2. Run the script:
```bash
python scene_splitter.py
```

## Output
- `scenes/` directory containing individual scene videos
- `scene_analysis.txt` with detailed scene information

## Configuration
Adjust these constants in `scene_splitter.py`:
- `FRAMES_TO_SKIP`: Frames to skip between analyses (default: 15)
- `FRAMES_PER_BATCH`: Frames per API call (default: 4)
- `threshold` in `detect_scene_change()`: Scene change sensitivity (default: 30.0)

## How It Works
1. Extracts frames at regular intervals
2. Analyzes frames using GPT-4 Vision for high-level scene changes
3. Uses OpenCV for low-level motion detection
4. Combines both analyses for accurate scene detection
5. Splits video at detected scene changes

## Notes
- API costs depend on video length and frame analysis frequency
- Processing time varies with video length
- Requires sufficient storage for temporary frames and output videos 