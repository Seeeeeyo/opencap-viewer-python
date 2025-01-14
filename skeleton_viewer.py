import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QLabel
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pathlib import Path
import json
import trimesh
from dataclasses import dataclass
import time
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage


@dataclass
class Body:
    translation: np.ndarray
    rotation: np.ndarray
    scale_factors: np.ndarray
    attached_geometries: list


class SkeletonViewer:
    def __init__(self, width=1280, height=720, target_fps=30):
        self.width = width
        self.height = height
        self.target_fps = target_fps  # Store target FPS
        self.app = QApplication([])

        # Create main window
        self.window = QWidget()
        self.window.resize(1400, 800)  # Set larger initial size
        self.layout = QVBoxLayout()
        self.window.setLayout(self.layout)

        # Create OpenGL view
        self.view = gl.GLViewWidget()
        self.view.resize(width, height)
        self.layout.addWidget(self.view)

        # Create control buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Step backward button
        self.step_back_button = QPushButton("←")
        self.step_back_button.setFixedHeight(30)  # Set fixed height for buttons
        self.step_back_button.clicked.connect(self.step_backward)
        self.button_layout.addWidget(self.step_back_button)
        
        # Play/Pause button
        self.play_button = QPushButton("Pause")
        self.play_button.setFixedHeight(30)
        self.play_button.clicked.connect(self.toggle_play)
        self.button_layout.addWidget(self.play_button)
        
        # Step forward button
        self.step_forward_button = QPushButton("→")
        self.step_forward_button.setFixedHeight(30)
        self.step_forward_button.clicked.connect(self.step_forward)
        self.button_layout.addWidget(self.step_forward_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedHeight(30)
        self.reset_button.clicked.connect(self.reset_animation)
        self.button_layout.addWidget(self.reset_button)
        
        # Record button
        self.record_button = QPushButton("Record")
        self.record_button.setFixedHeight(30)
        self.record_button.clicked.connect(self.toggle_recording)
        self.button_layout.addWidget(self.record_button)
        
        # Recording label
        self.recording_label = QLabel("Recording")
        self.recording_label.setFixedHeight(15)  # Match button height
        self.recording_label.setStyleSheet("color: red; font-size: 16px; padding: 0px;")
        self.recording_label.setAlignment(Qt.AlignCenter)  # Center text vertically and horizontally
        self.recording_label.hide()  # Initially hidden
        self.button_layout.addWidget(self.recording_label)
        
        self.layout.addLayout(self.button_layout)

        self.window.show()

        # Set background color to dark gray for better contrast
        self.view.setBackgroundColor((40, 40, 40))

        # Set camera position for better initial view
        self.view.setCameraPosition(distance=3, elevation=30, azimuth=0)
        self.view.opts['center'] = pg.Vector(0, 1, 0)

        # Add coordinate system orientation fix
        self.base_transform = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Add coordinate system reference
        # Move reference to bottom-left corner and add legend
        ref_height = 1.5  # Height above ground
        arrow_length = 0.5  # Arrow length
        arrow_width = 5    # Line thickness
        ref_offset = np.array([-8, -8, 0])  # Offset to bottom-left corner
        
        # X axis - Red
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, ref_height], [arrow_length, 0, ref_height]]) + ref_offset, 
            color=(1, 0, 0, 1), 
            width=arrow_width,
            glOptions='additive'
        )
        self.view.addItem(x_axis)
        # Make X label larger and more visible
        x_label = gl.GLTextItem(
            pos=ref_offset + np.array([arrow_length + 0.1, 0, ref_height]),
            text='X',
            color=(1, 0, 0, 1),
            font=pg.QtGui.QFont('Helvetica', 16)
        )
        self.view.addItem(x_label)
        
        # Y axis - Green
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, ref_height], [0, arrow_length, ref_height]]) + ref_offset, 
            color=(0, 1, 0, 1), 
            width=arrow_width,
            glOptions='additive'
        )
        self.view.addItem(y_axis)
        # Make Y label larger and more visible
        y_label = gl.GLTextItem(
            pos=ref_offset + np.array([0, arrow_length + 0.1, ref_height]),
            text='Y',
            color=(0, 1, 0, 1),
            font=pg.QtGui.QFont('Helvetica', 16)
        )
        self.view.addItem(y_label)
        
        # Z axis - Blue
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, ref_height], [0, 0, arrow_length + ref_height]]) + ref_offset, 
            color=(0, 0, 1, 1), 
            width=arrow_width,
            glOptions='additive'
        )
        self.view.addItem(z_axis)
        # Make Z label larger and more visible
        z_label = gl.GLTextItem(
            pos=ref_offset + np.array([0, 0, arrow_length + 0.2 + ref_height]),
            text='Z',
            color=(0, 0, 1, 1),
            font=pg.QtGui.QFont('Helvetica', 16)
        )
        self.view.addItem(z_label)
        
        # Add legend text with larger font
        legend_text = gl.GLTextItem(
            pos=ref_offset + np.array([0.7, -0.5, ref_height]),
            text='Reference Frame:\nX: Forward/Back (Red)\nY: Left/Right (Green)\nZ: Up/Down (Blue)',
            color=(1, 1, 1, 1),
            font=pg.QtGui.QFont('Helvetica', 12)
        )

        # Add floor plane and grid
        # Create floor plane vertices and faces
        floor_verts = np.array([
            [-10, -10, 0], # Bottom left
            [10, -10, 0],  # Bottom right
            [10, 10, 0],   # Top right
            [-10, 10, 0]   # Top left
        ], dtype=np.float32)
        
        floor_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.uint32)
        
        # Create floor plane
        floor = gl.GLMeshItem(
            vertexes=floor_verts,
            faces=floor_faces,
            smooth=False,
            shader='balloon',
            color=(0.3, 0.3, 0.3, 1.0)  # Dark gray, fully opaque
        )
        self.view.addItem(floor)

        # Add grid on top of the floor
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20)
        grid.setSpacing(x=1, y=1)
        grid.translate(0, 0, 0.01)  # Slightly above the floor to prevent z-fighting
        grid.setColor((0.7, 0.7, 0.7, 0.3))  # Light gray, semi-transparent
        self.view.addItem(grid)

        # Animation data - modified to handle multiple skeletons
        self.frames = []  # Will store the longest frame sequence
        self.current_frame = 0
        self.skeletons = []  # List to store data for each skeleton
        self.mesh_items = {}
        self.frame_buffer = {}  # Store frame data

        # Recording state
        self.is_recording = False
        self.video_writer = None

        # Performance optimization
        self.last_time = time.time()
        self.frame_time = 1.0 / target_fps  # Use target FPS for frame timing
        self.is_playing = True  # Animation state

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("Play" if not self.is_playing else "Pause")

    def reset_animation(self):
        self.current_frame = 0
        self.is_playing = False
        self.play_button.setText("Play")

        # Immediately update the view to show the first frame
        for key, mesh_item in self.mesh_items.items():
            vertices = self.frame_buffer[key]['vertices'][self.current_frame]
            mesh_item.setMeshData(vertexes=vertices)

    def load_animations(self, json_paths):
        for json_path in json_paths:
            print(f"Loading animation from {json_path}...")
            self.load_animation(json_path)
            print(f"Finished loading {json_path}")

    def load_animation(self, json_path, color=(1.0, 1.0, 1.0, 1.0)):
        print(f"Loading animation data from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create unique identifier for this skeleton
        skeleton_id = len(self.skeletons)
        
        # Update frames to be the longest animation
        if len(data['time']) > len(self.frames):
            self.frames = data['time']

        # Store skeleton data
        skeleton_data = {
            'id': skeleton_id,
            'frames': data['time'],
            'bodies': {},
            'color': color
        }

        # Pre-allocate frame buffer
        print("Pre-allocating frame buffer...")
        for body_name, body_data in data['bodies'].items():
            translation = np.array(body_data['translation'])
            rotation = np.array(body_data['rotation'])
            scale = np.array(body_data['scaleFactors'])

            skeleton_data['bodies'][body_name] = Body(
                translation=translation,
                rotation=rotation,
                scale_factors=scale,
                attached_geometries=body_data['attachedGeometries']
            )

            for geom in body_data['attachedGeometries']:
                obj_name = geom.replace('.vtp', '.obj')
                mesh_path = Path("public/dataForVisualizer/geometry-obj") / obj_name

                if mesh_path.exists():
                    mesh = trimesh.load(str(mesh_path))
                    vertices = np.array(mesh.vertices, dtype=np.float32)
                    faces = np.array(mesh.faces, dtype=np.uint32)

                    # Create mesh item with basic stable rendering
                    mesh_item = gl.GLMeshItem(
                        vertexes=vertices,
                        faces=faces,
                        smooth=False,  # Disable smooth shading for stability
                        computeNormals=False,  # Disable normal computation
                        shader='balloon',  # Use basic shader
                        color=color,
                        drawEdges=False,  # Disable edges for stability
                        glOptions='opaque'  # Basic rendering mode
                    )

                    mesh_item.initializeGL()

                    key = f"skeleton_{skeleton_id}_{body_name}{geom}"
                    self.mesh_items[key] = mesh_item
                    self.view.addItem(mesh_item)

                    # Pre-allocate buffer for this mesh
                    self.frame_buffer[key] = {
                        'original_vertices': vertices,
                        'vertices': np.zeros((len(data['time']), len(vertices), 3), dtype=np.float32),
                        'faces': faces
                    }

        self.skeletons.append(skeleton_data)

        # Compute frame data in batches
        print("Computing frame data...")
        batch_size = 100
        total_frames = len(data['time'])
        for start_idx in range(0, total_frames, batch_size):
            end_idx = min(start_idx + batch_size, total_frames)
            self._compute_frame_batch(skeleton_id, start_idx, end_idx)
            print(f"Processed frames {start_idx} to {end_idx}")

    def _compute_frame_batch(self, skeleton_id, start_idx, end_idx):
        skeleton_data = self.skeletons[skeleton_id]
        for frame_idx in range(start_idx, end_idx):
            for body_name, body in skeleton_data['bodies'].items():
                transform = self._compute_transform(
                    body.translation[frame_idx],
                    body.rotation[frame_idx],
                    body.scale_factors,
                    skeleton_id
                )

                for geom in body.attached_geometries:
                    key = f"skeleton_{skeleton_id}_{body_name}{geom}"
                    mesh_item = self.mesh_items[key]
                    orig_verts = self.frame_buffer[key]['original_vertices']

                    # Apply transform
                    verts = np.ones((len(orig_verts), 4), dtype=np.float32)
                    verts[:, :3] = orig_verts
                    transformed = (verts @ transform.T)[:, :3]

                    # Store in buffer
                    self.frame_buffer[key]['vertices'][frame_idx] = transformed

    def _compute_transform(self, translation, rotation, scale, skeleton_id=0):
        transform = np.eye(4, dtype=np.float32)

        # Scale
        transform[:3, :3] = np.diag(scale)

        # Rotation
        rot_mat = np.array([
            [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
            [np.sin(rotation[2]), np.cos(rotation[2]), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.cos(rotation[1]), 0, np.sin(rotation[1])],
            [0, 1, 0],
            [-np.sin(rotation[1]), 0, np.cos(rotation[1])]
        ]) @ np.array([
            [1, 0, 0],
            [0, np.cos(rotation[0]), -np.sin(rotation[0])],
            [0, np.sin(rotation[0]), np.cos(rotation[0])]
        ])

        transform[:3, :3] = transform[:3, :3] @ rot_mat
        
        # Add offset based on skeleton ID (1.5 units apart on Z axis)
        offset = np.array([0, 0, skeleton_id * 1.5])
        transform[:3, 3] = translation + offset

        return self.base_transform @ transform

    def step_forward(self):
        # Pause animation if playing
        self.is_playing = False
        self.play_button.setText("Play")

        # Move to next frame
        self.current_frame = (self.current_frame + 1) % len(self.frames)

        # Update view
        self.update_frame_display()

    def step_backward(self):
        # Pause animation if playing
        self.is_playing = False
        self.play_button.setText("Play")

        # Move to previous frame
        self.current_frame = (self.current_frame - 1) % len(self.frames)

        # Update view
        self.update_frame_display()

    def update_frame_display(self):
        # Update meshes for current frame
        for skeleton in self.skeletons:
            skeleton_id = skeleton['id']
            frame_idx = self.current_frame % len(skeleton['frames'])
            
            for body_name, body in skeleton['bodies'].items():
                for geom in body.attached_geometries:
                    key = f"skeleton_{skeleton_id}_{body_name}{geom}"
                    if key in self.mesh_items and key in self.frame_buffer:
                        vertices = self.frame_buffer[key]['vertices'][frame_idx]
                        self.mesh_items[key].setMeshData(vertexes=vertices)

    def toggle_recording(self):
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.recording_label.show()  # Show recording label
            
            # Create video writer with H264 codec
            output_path = 'skeleton_animation.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            self.video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                30,  # FPS
                (self.width, self.height)
            )
            if not self.video_writer.isOpened():
                print("Error: VideoWriter not opened")
                # Try fallback codec
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                self.video_writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    30,
                    (self.width, self.height)
                )
            print(f"Started recording to {output_path}")
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setText("Record")
            self.recording_label.hide()  # Hide recording label
            if self.video_writer is not None:
                self.video_writer.release()
                cv2.destroyAllWindows()  # Clean up any OpenCV windows
                self.video_writer = None
                print("Recording saved and finalized")

    def capture_frame(self):
        if self.is_recording and self.video_writer is not None:
            try:
                # Capture the current view using renderToArray
                frame = self.view.renderToArray((self.width, self.height))
                
                # Convert to BGR format for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                self.video_writer.write(frame)
                
                # Save a debug frame occasionally
                # if self.current_frame % 30 == 0:  # Save every 30th frame
                #    cv2.imwrite(f'debug_frame_{self.current_frame}.png', frame)
                
            except Exception as e:
                print(f"Error capturing frame: {e}")

    def update(self):
        if self.is_playing:
            current_time = time.time()
            
            # Calculate time-based frame advancement
            if not self.is_recording:
                # Normal playback - use real time
                delta = current_time - self.last_time
                frames_to_advance = max(1, int(delta * self.target_fps))
            else:
                # Recording - advance exactly one frame per update
                frames_to_advance = 1
            
            self.last_time = current_time

            # Update meshes from pre-computed buffer
            self.update_frame_display()

            # Capture frame if recording
            if self.is_recording:
                self.capture_frame()

            # Advance frame(s)
            self.current_frame = (self.current_frame + frames_to_advance) % len(self.frames)

        # Schedule next update
        # Use shorter interval during recording to maintain smooth capture
        update_interval = int(1000 / self.target_fps)  # Convert FPS to milliseconds
        pg.QtCore.QTimer.singleShot(update_interval, self.update)

    def run(self):
        # Start animation
        self.last_time = time.time()
        self.update()

        # Start Qt event loop
        self.app.exec_()


if __name__ == "__main__":
    viewer = SkeletonViewer()
    viewer.run()