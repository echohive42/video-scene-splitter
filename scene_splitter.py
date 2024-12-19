import cv2
import os
from openai import OpenAI
import base64
from termcolor import colored
import numpy as np
from datetime import timedelta
from difflib import SequenceMatcher

# Constants
FRAMES_TO_SKIP = 15  # Reduced to catch more subtle changes
FRAMES_PER_BATCH = 4
VIDEO_PATH = "video.mp4"
API_KEY = os.getenv("OPENAI_API_KEY")
TEMP_FRAME_DIR = "temp_frames"
OUTPUT_DIR = "scenes"
MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.7  # Threshold for scene similarity

# Create necessary directories
os.makedirs(TEMP_FRAME_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    client = OpenAI(api_key=API_KEY)
    print(colored("OpenAI client initialized successfully", "green"))
except Exception as e:
    print(colored(f"Error initializing OpenAI client: {str(e)}", "red"))
    exit(1)

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(colored(f"Error encoding image: {str(e)}", "red"))
        return None

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def detect_scene_change(frame1, frame2, threshold=30.0):
    """Detect scene change using frame difference"""
    try:
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        score = np.mean(np.abs(gray1 - gray2))
        return score > threshold
        
    except Exception as e:
        print(colored(f"Error in scene change detection: {str(e)}", "red"))
        return False

def analyze_frames_batch(frame_paths, timestamps):
    try:
        content = [
            {
                "type": "text",
                "text": "Analyze these consecutive frames for scene changes. A scene change occurs when there is:"
                        "\n1. A different location or setting"
                        "\n2. A significant change in camera angle (not minor movements)"
                        "\n3. A completely different action or subject"
                        "\n4. A major lighting change"
                        "\nMinor changes in the same scene should NOT be counted as scene changes."
                        "\nReturn ONLY frame numbers where definite scene changes occur, or 'None' if no changes."
                        "\nFormat: Scene changes: [frame numbers or None]"
            }
        ]
        
        # Load frames for OpenCV analysis
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frames.append(frame)
            frame_base64 = encode_image_to_base64(frame_path)
            if frame_base64:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}"
                    }
                })
        
        # Get GPT's analysis
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=100  # Reduced since we only need scene change numbers
        )
        gpt_analysis = response.choices[0].message.content

        # Combine GPT analysis with OpenCV detection
        scene_changes = set()
        
        # Parse GPT's scene changes
        if "Scene changes:" in gpt_analysis:
            changes_line = gpt_analysis.split("Scene changes:")[-1].strip()
            if changes_line.lower() != "none":
                gpt_changes = [int(x.strip()) for x in changes_line.split(',') if x.strip().isdigit()]
                scene_changes.update(gpt_changes)
        
        # Add OpenCV-detected changes
        for i in range(len(frames)-1):
            if detect_scene_change(frames[i], frames[i+1]):
                scene_changes.add(i+1)
        
        # Sort the combined scene changes
        scene_changes = sorted(list(scene_changes))
        
        return f"Scene changes: {scene_changes if scene_changes else 'None'}"
    except Exception as e:
        print(colored(f"Error analyzing frames with GPT-4 Vision: {str(e)}", "red"))
        return None

def extract_scene(video_capture, start_frame, end_frame, scene_number):
    try:
        output_path = os.path.join(OUTPUT_DIR, f"scene_{scene_number}.mp4")
        
        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set frame position to start
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(colored(f"Extracting scene {scene_number} (frames {start_frame} to {end_frame})", "cyan"))
        
        # Read and write frames
        for _ in range(end_frame - start_frame):
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(colored(f"Scene {scene_number} saved to {output_path}", "green"))
        return output_path
    except Exception as e:
        print(colored(f"Error extracting scene: {str(e)}", "red"))
        return None

def main():
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise Exception("Error opening video file")
        
        print(colored("Successfully opened video file", "green"))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scenes = []
        frame_count = 0
        batch_frames = []
        batch_timestamps = []
        current_scene_start = 0
        scene_number = 1
        
        print(colored("Starting scene analysis...", "cyan"))
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % FRAMES_TO_SKIP == 0:
                frame_path = os.path.join(TEMP_FRAME_DIR, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                timestamp = timedelta(seconds=frame_count/fps)
                
                batch_frames.append(frame_path)
                batch_timestamps.append(timestamp)
                
                if len(batch_frames) == FRAMES_PER_BATCH:
                    print(colored(f"Analyzing batch at timestamp {timestamp}", "cyan"))
                    analysis = analyze_frames_batch(batch_frames, batch_timestamps)
                    
                    if analysis:
                        # Extract scene changes from analysis
                        scene_changes = []
                        if "Scene changes:" in analysis:
                            changes_line = analysis.split("Scene changes:")[-1].strip()
                            if changes_line.lower() != "none":
                                scene_changes = [int(x.strip()) for x in changes_line.split(',') if x.strip().isdigit()]
                        
                        if scene_changes:
                            for change in scene_changes:
                                # Calculate actual frame number for the scene change
                                scene_frame = frame_count - (FRAMES_PER_BATCH - change) * FRAMES_TO_SKIP
                                
                                # Extract the scene
                                scene_path = extract_scene(cap, current_scene_start, scene_frame, scene_number)
                                
                                scenes.append({
                                    'scene_number': scene_number,
                                    'start_frame': current_scene_start,
                                    'end_frame': scene_frame,
                                    'start_time': str(timedelta(seconds=current_scene_start/fps)),
                                    'end_time': str(timedelta(seconds=scene_frame/fps)),
                                    'video_path': scene_path,
                                    'description': analysis
                                })
                                
                                current_scene_start = scene_frame
                                scene_number += 1
                        
                        print(colored(f"Batch analysis: {analysis}", "yellow"))
                    
                    # Clean up batch frames
                    for f in batch_frames:
                        if os.path.exists(f):
                            os.remove(f)
                    batch_frames = []
                    batch_timestamps = []
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(colored(f"Processed {frame_count}/{total_frames} frames", "cyan"))
        
        # Extract final scene if needed
        if current_scene_start < total_frames - 1:
            scene_path = extract_scene(cap, current_scene_start, total_frames - 1, scene_number)
            scenes.append({
                'scene_number': scene_number,
                'start_frame': current_scene_start,
                'end_frame': total_frames - 1,
                'start_time': str(timedelta(seconds=current_scene_start/fps)),
                'end_time': str(timedelta(seconds=(total_frames - 1)/fps)),
                'video_path': scene_path,
                'description': "Final scene"
            })
        
        # Clean up
        cap.release()
        if os.path.exists(TEMP_FRAME_DIR):
            for f in os.listdir(TEMP_FRAME_DIR):
                os.remove(os.path.join(TEMP_FRAME_DIR, f))
            os.rmdir(TEMP_FRAME_DIR)
        
        # Save detailed analysis
        try:
            with open('scene_analysis.txt', 'w', encoding='utf-8') as f:
                for scene in scenes:
                    f.write(f"Scene {scene['scene_number']}:\n")
                    f.write(f"Time Range: {scene['start_time']} - {scene['end_time']}\n")
                    f.write(f"Frame Range: {scene['start_frame']} - {scene['end_frame']}\n")
                    f.write(f"Video File: {scene['video_path']}\n")
                    f.write(f"Analysis:\n{scene['description']}\n")
                    f.write("-" * 50 + "\n")
            print(colored("Scene analysis saved to scene_analysis.txt", "green"))
        except Exception as e:
            print(colored(f"Error saving scene analysis: {str(e)}", "red"))
            
    except Exception as e:
        print(colored(f"An error occurred: {str(e)}", "red"))
        if 'cap' in locals():
            cap.release()
        if os.path.exists(TEMP_FRAME_DIR):
            for f in os.listdir(TEMP_FRAME_DIR):
                os.remove(os.path.join(TEMP_FRAME_DIR, f))
            os.rmdir(TEMP_FRAME_DIR)

if __name__ == "__main__":
    main() 