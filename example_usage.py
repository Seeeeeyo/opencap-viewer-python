from skeleton_viewer import SkeletonViewer
from pathlib import Path

# Create viewer instance with desired FPS
viewer = SkeletonViewer(width=1280, height=720, target_fps=30)

# Load animations with different colors
json_paths = [
    (Path("test.json"), (1.0, 0.0, 0.0, 1.0)),  # Red
    (Path("test.json"), (0.0, 1.0, 0.0, 1.0)),  # Green
    #(Path("public/dataForVisualizer/visualizerTransforms.json"), (0.0, 0.0, 1.0, 1.0))   # Blue
]

for json_path, color in json_paths:
    viewer.load_animation(json_path, color)

# Run viewer
viewer.run() 