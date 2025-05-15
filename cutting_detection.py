import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
import os
from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise
import argparse

class DroneExtendedKalmanFilter:
    """
    Extended Kalman Filter for drone tracking with non-linear motion model.
    State vector: [x, y, vx, vy, ax, ay]
    - x, y: position
    - vx, vy: velocity
    - ax, ay: acceleration
    """
    def __init__(self, pixel_to_meter_ratio=None, max_height_meters=4.0):
        # Create EKF with 6 state variables and 2 measurement variables
        self.ekf = EKF(dim_x=6, dim_z=2)
        
        # State transition matrix (will be updated in predict function with dt)
        self.ekf.F = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement function (we only measure x, y positions)
        self.ekf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Initial state covariance
        self.ekf.P = np.eye(6) * 10
        
        # Measurement noise covariance
        self.ekf.R = np.eye(2) * 5.0  # Adjusted for pixel measurements
        
        # Process noise covariance (will be updated in predict with dt)
        self.ekf.Q = np.eye(6) * 0.1
        
        # Initial state
        self.ekf.x = np.zeros((6, 1))
        
        # Time since last update
        self.last_time = None
        self.initialized = False
        
        # Variables for tracking the detection history
        self.detection_history = []
        self.frames_since_last_detection = 0
        self.max_frames_to_track = 30  # Maximum frames to keep tracking without detection
        
        # Detection confidence
        self.confidence = 0.0
        
        # Velocity and direction smoothing (increased window sizes)
        self.velocity_history = []
        self.direction_history = []
        self.velocity_window = 15  # Increased from 10 for better smoothing
        self.direction_window = 15  # Increased from 10 for better smoothing
        
        # For calculating actual velocity in meters
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.max_height_meters = max_height_meters
        
        # For more aggressive velocity smoothing
        self.raw_velocity_values = []
        self.smoothing_window = 20  # Large window for aggressive smoothing
    
    def set_pixel_to_meter_ratio(self, frame_height):
        """
        Set the pixels to meters ratio based on the max height of the drone
        """
        self.pixel_to_meter_ratio = frame_height / self.max_height_meters
    
    def HJacobian(self, x):
        """
        Compute Jacobian of H matrix for EKF
        For linear measurements, this is just H
        """
        return self.ekf.H
    
    def Hx(self, x):
        """
        Measurement function for EKF
        For linear measurements, this is just H*x
        """
        return self.ekf.H @ x
    
    def predict(self, dt):
        """
        Predict next state based on constant acceleration model
        """
        # Update state transition matrix with dt
        self.ekf.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Further reduced process noise for smoother tracking
        q = Q_discrete_white_noise(dim=2, dt=dt, var=0.003)  # Reduced from 0.005
        q_vel = Q_discrete_white_noise(dim=2, dt=dt, var=0.008)  # Reduced from 0.01
        q_acc = Q_discrete_white_noise(dim=2, dt=dt, var=0.015)  # Reduced from 0.02
        self.ekf.Q = block_diag(q, q_vel, q_acc)
        
        self.ekf.predict()
        self.frames_since_last_detection += 1
        
        # Return predicted position
        return (self.ekf.x[0, 0], self.ekf.x[1, 0])
    
    def update(self, cx=None, cy=None, confidence=0.0, dt=1.0/30.0):
        """
        Update the filter with a new measurement
        If no measurement, just predict
        """
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        if self.last_time is not None:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        # If we have a detection
        if cx is not None and cy is not None:
            z = np.array([[cx], [cy]], dtype=np.float32)
            
            # Initialize the filter if this is the first detection
            if not self.initialized:
                self.ekf.x = np.array([[cx], [cy], [0], [0], [0], [0]], dtype=np.float32)
                self.initialized = True
            else:
                # Update the filter with the measurement
                self.ekf.update(z, HJacobian=self.HJacobian, Hx=self.Hx)
            
            self.frames_since_last_detection = 0
            self.confidence = confidence
            
            # Store detection in history (limited size)
            self.detection_history.append((cx, cy))
            if len(self.detection_history) > 10:
                self.detection_history.pop(0)
        
        # If we've lost track for too long, reinitialize next time
        if self.frames_since_last_detection > self.max_frames_to_track:
            self.initialized = False
            
        # Predict next state
        return self.predict(dt)
    
    def get_state(self):
        """
        Return the current state estimates
        """
        if not self.initialized:
            return None
            
        return {
            'x': self.ekf.x[0, 0],
            'y': self.ekf.x[1, 0],
            'vx': self.ekf.x[2, 0],
            'vy': self.ekf.x[3, 0],
            'ax': self.ekf.x[4, 0],
            'ay': self.ekf.x[5, 0],
            'confidence': self.confidence
        }
    
    def get_smoothed_velocity(self, fps=30.0):
        """
        Get smoothed velocity and direction from Kalman filter state
        Also calculates actual velocity in meters per second
        """
        if not self.initialized:
            return None
        
        # Get current velocity from state
        vx = self.ekf.x[2, 0]
        vy = self.ekf.x[3, 0]
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        # Store raw velocity for aggressive smoothing
        self.raw_velocity_values.append(velocity_magnitude)
        if len(self.raw_velocity_values) > self.smoothing_window:
            self.raw_velocity_values.pop(0)
        
        # Apply Gaussian-weighted smoothing for velocity
        if len(self.raw_velocity_values) >= 5:  # Need at least 5 values for good smoothing
            weights = np.exp(-0.5 * np.linspace(-2, 2, len(self.raw_velocity_values))**2)
            weights = weights / np.sum(weights)  # Normalize weights
            smoothed_velocity = np.sum(np.array(self.raw_velocity_values) * weights)
        else:
            smoothed_velocity = velocity_magnitude
            
        # Calculate direction angle (negative vy since y increases downward in images)
        direction_angle = np.degrees(np.arctan2(-vy, vx))
        if direction_angle < 0:
            direction_angle += 360
            
        # Store in history for additional smoothing
        self.velocity_history.append(smoothed_velocity)
        self.direction_history.append(direction_angle)
        
        # Keep history at fixed length
        if len(self.velocity_history) > self.velocity_window:
            self.velocity_history.pop(0)
        if len(self.direction_history) > self.direction_window:
            self.direction_history.pop(0)
        
        # Apply double smoothing to velocity (two-stage smoothing)
        # First stage: weighted moving average
        weights = np.linspace(0.5, 1.0, len(self.velocity_history))
        weights = weights / np.sum(weights)  # Normalize weights
        twice_smoothed_velocity = np.sum(np.array(self.velocity_history) * weights)
        
        # Special handling for direction smoothing due to circular nature
        # Use a Gaussian-weighted circular mean
        weights = np.exp(-0.5 * np.linspace(-2, 2, len(self.direction_history))**2)
        weights = weights / np.sum(weights)  # Normalize weights
        
        sin_sum = 0
        cos_sum = 0
        for i, angle in enumerate(self.direction_history):
            rad = np.radians(angle)
            sin_sum += np.sin(rad) * weights[i]
            cos_sum += np.cos(rad) * weights[i]
        
        avg_angle = np.degrees(np.arctan2(sin_sum, cos_sum))
        if avg_angle < 0:
            avg_angle += 360
            
        # Convert angle to 8-direction compass point
        directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        direction_index = int(((avg_angle + 22.5) % 360) / 45)
        direction_str = directions[direction_index]
        
        # Calculate actual velocity in meters per second
        actual_velocity_mps = None
        if self.pixel_to_meter_ratio and fps > 0:
            # Convert from pixels/frame to meters/second
            actual_velocity_mps = twice_smoothed_velocity / self.pixel_to_meter_ratio * fps
        
        return {
            'vx': vx,
            'vy': vy,
            'velocity': twice_smoothed_velocity,
            'direction_angle': avg_angle,
            'direction': direction_str,
            'actual_velocity_mps': actual_velocity_mps
        }
    
    def get_bbox(self, box_size=100):
        """
        Get the bounding box from current state
        """
        if not self.initialized:
            return None
            
        cx, cy = self.ekf.x[0, 0], self.ekf.x[1, 0]
        return (
            int(cx - box_size / 2),
            int(cy - box_size / 2),
            int(cx + box_size / 2),
            int(cy + box_size / 2)
        )
    
    def estimate_box_size(self, bbox):
        """
        Estimate appropriate box size from detection
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(width, height)
    
    
class DroneTracker:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5, drone_class_id=0, 
                max_height_meters=4.0, fps=30.0):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.drone_class_id = drone_class_id
        self.ekf = DroneExtendedKalmanFilter(max_height_meters=max_height_meters)
        self.tracking_data = []  # For storing tracking results
        self.frame_counter = 0
        self.box_size = 100  # Default box size, will be adaptive
        self.fps = fps
        
        # Store previous raw detection for velocity calculation
        self.prev_detection = None
        self.prev_frame_id = None
        
        # Define the directions list at class level
        self.directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        
        # Track velocity and direction for better visualization
        self.velocity_trail = []
        self.trail_length = 60  # Store last 60 positions (about 2 seconds at 30fps)
        
        # Height estimation variables
        self.max_height_meters = max_height_meters
        self.pixel_to_meter_ratio = None
        
    def _select_best_detection(self, detections):
        """
        Select the best detection among multiple candidates
        Strategy: Use highest confidence detection, or closest to previous position
        """
        if not detections:
            return None, 0.0
            
        if len(detections) == 1:
            # Return position and confidence for a single detection
            cx, cy, conf = detections[0]
            return (cx, cy), conf
            
        # Get current state
        state = self.ekf.get_state()
        
        if state is None:
            # If no previous state, choose highest confidence
            best_det = max(detections, key=lambda d: d[2])
            cx, cy, conf = best_det
            return (cx, cy), conf
        
        # If we have a previous state, use a weighted combination of confidence and distance
        best_score = -float('inf')
        best_detection = None
        best_conf = 0.0
        
        current_pos = np.array([state['x'], state['y']])
        
        for det in detections:
            cx, cy, conf = det
            det_pos = np.array([cx, cy])
            distance = np.linalg.norm(det_pos - current_pos)
            
            # Convert distance to a score (closer = higher score)
            # Max reasonable distance is 25% of frame diagonal
            max_dist = 1000  # Arbitrary large value
            distance_score = max(0, 1 - distance / max_dist)
            
            # Combined score (70% confidence, 30% proximity to last known position)
            score = 0.7 * conf + 0.3 * distance_score
            
            if score > best_score:
                best_score = score
                best_detection = (cx, cy)
                best_conf = conf
                
        return best_detection, best_conf
    
    def process_frame(self, frame, frame_id):
        """
        Process a single frame
        Returns the processed frame and tracking data
        """
        # Set pixel to meter ratio if not set
        if self.pixel_to_meter_ratio is None:
            frame_height = frame.shape[0]
            self.ekf.set_pixel_to_meter_ratio(frame_height)
            self.pixel_to_meter_ratio = frame_height / self.max_height_meters
        
        self.frame_counter = frame_id
        
        # Run YOLOv8 detection
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[self.drone_class_id],
            persist=True,
            stream=True
        )
        
        # Process detections and select the best one
        detections = []
        tracked_id = None
        
        for r in results:
            if hasattr(r, 'boxes') and r.boxes and len(r.boxes.xyxy) > 0:
                # Process all drone detections
                for i in range(len(r.boxes.xyxy)):
                    xyxy = r.boxes.xyxy[i].cpu().numpy()
                    conf = float(r.boxes.conf[i].cpu().numpy()) if hasattr(r.boxes, 'conf') else 0.0
                    
                    # Get tracking ID if available
                    if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                        track_id = int(r.boxes.id[i].cpu().numpy())
                        tracked_id = track_id
                    
                    cx = float((xyxy[0] + xyxy[2]) / 2)
                    cy = float((xyxy[1] + xyxy[3]) / 2)
                    
                    # Update box size based on detection
                    self.box_size = int(0.2 * self.box_size + 0.8 * (max(xyxy[2]-xyxy[0], xyxy[3]-xyxy[1])))
                    
                    detections.append((cx, cy, conf, xyxy))
        
        # Select the best detection
        selected_detection = None
        confidence = 0.0
        detected_bbox = None
        
        if detections:
            (cx, cy), confidence = self._select_best_detection([(d[0], d[1], d[2]) for d in detections])
            selected_detection = (cx, cy)
            
            # Find the corresponding bbox
            for det in detections:
                if det[0] == cx and det[1] == cy:
                    detected_bbox = det[3]
                    break
        
        # Update the Kalman filter with detection (or just predict if no detection)
        smoothed_cx, smoothed_cy = self.ekf.update(
            cx=selected_detection[0] if selected_detection else None,
            cy=selected_detection[1] if selected_detection else None,
            confidence=confidence
        )
        
        # Get current state from Kalman filter
        state = self.ekf.get_state()
        
        # Get smoothed velocity and direction from Kalman filter
        kalman_velocity_data = self.ekf.get_smoothed_velocity(fps=self.fps) if state is not None else None
        
        # Update velocity trail for visualization
        if state is not None:
            self.velocity_trail.append((int(smoothed_cx), int(smoothed_cy)))
            if len(self.velocity_trail) > self.trail_length:
                self.velocity_trail.pop(0)
        
        # Draw results on the frame
        if state is not None:
            # Draw smoothed bounding box
            if self.box_size <= 0:
                self.box_size = 100  # Fallback if box size calculation failed
                
            x1, y1, x2, y2 = self.ekf.get_bbox(box_size=self.box_size)
            
            # Ensure box is within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Draw the smoothed bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw the motion trail
            if len(self.velocity_trail) > 1:
                for i in range(1, len(self.velocity_trail)):
                    # Color fades from blue to red based on recency
                    alpha = i / len(self.velocity_trail)
                    color = (
                        int(255 * (1 - alpha)),  # Blue component
                        0,                      # Green component
                        int(255 * alpha)        # Red component
                    )
                    cv2.line(frame, self.velocity_trail[i-1], self.velocity_trail[i], color, 2)
            
            # Draw Kalman velocity vector if available
            if kalman_velocity_data:
                start_point = (int(smoothed_cx), int(smoothed_cy))
                # Scale the vector for better visualization
                vector_scale = 20  # Making it larger for better visibility
                end_point = (
                    int(smoothed_cx + kalman_velocity_data['vx'] * vector_scale), 
                    int(smoothed_cy + kalman_velocity_data['vy'] * vector_scale)
                )
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), 3)  # Green arrow for Kalman vector
                
                # Create large text at top right of frame for velocity metrics
                vel_text = f"Velocity: {kalman_velocity_data['velocity']:.1f} px/frame"
                
                # Add actual velocity in m/s if available
                actual_vel_text = ""
                if kalman_velocity_data['actual_velocity_mps'] is not None:
                    actual_vel_text = f"Actual Velocity: {kalman_velocity_data['actual_velocity_mps']:.2f} m/s"
                
                dir_text = f"Direction: {kalman_velocity_data['direction']} ({int(kalman_velocity_data['direction_angle'])}°)"
                conf_text = f"Confidence: {confidence:.2f}"
                
                # Calculate height from ground (y-position)
                # Assuming bottom of frame is ground level and top is maximum height
                height_ratio = 1.0 - (smoothed_cy / frame.shape[0])
                height_meters = height_ratio * self.max_height_meters
                height_text = f"Height: {height_meters:.2f} m"
                
                # Parameters for large top-right text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                color = (0, 255, 255)  # Yellow
                
                # Calculate text positions (top right with some padding)
                text_width, text_height = cv2.getTextSize(vel_text, font, font_scale, thickness)[0]
                text_x = frame.shape[1] - text_width - 20  # 20 pixels from right edge
                
                # Draw background rectangle for better visibility
                cv2.rectangle(frame, 
                             (frame.shape[1] - text_width - 30, 10), 
                             (frame.shape[1] - 10, 30 + 5 * text_height + 60),
                             (0, 0, 0), -1)  # Black background
                
                # Draw text
                y_offset = 40
                cv2.putText(frame, vel_text, (text_x, y_offset), font, font_scale, color, thickness)
                y_offset += text_height + 10
                
                if actual_vel_text:
                    cv2.putText(frame, actual_vel_text, (text_x, y_offset), font, font_scale, color, thickness)
                    y_offset += text_height + 10
                
                cv2.putText(frame, dir_text, (text_x, y_offset), font, font_scale, color, thickness)
                y_offset += text_height + 10
                
                cv2.putText(frame, height_text, (text_x, y_offset), font, font_scale, color, thickness)
                y_offset += text_height + 10
                
                cv2.putText(frame, conf_text, (text_x, y_offset), font, font_scale, color, thickness)
            
            # Add tracking info near the bounding box
            info_text = f"Drone ID: {tracked_id if tracked_id is not None else 'N/A'}"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Store tracking data with direction info
            tracking_entry = {
                'frame_id': frame_id,
                'x': smoothed_cx,
                'y': smoothed_cy,
                'ekf_vx': state['vx'],
                'ekf_vy': state['vy'],
                'ekf_ax': state['ax'],
                'ekf_ay': state['ay'],
                'confidence': confidence,
                'detected': selected_detection is not None,
                'track_id': tracked_id if tracked_id is not None else -1
            }
            
            # Calculate height in meters
            height_ratio = 1.0 - (smoothed_cy / frame.shape[0])
            height_meters = height_ratio * self.max_height_meters
            tracking_entry['height_meters'] = height_meters
            
            # Add Kalman-smoothed velocity data if available
            if kalman_velocity_data:
                tracking_entry.update({
                    'smoothed_vx': kalman_velocity_data['vx'],
                    'smoothed_vy': kalman_velocity_data['vy'],
                    'smoothed_velocity': kalman_velocity_data['velocity'],
                    'smoothed_direction_angle': kalman_velocity_data['direction_angle'],
                    'smoothed_direction': kalman_velocity_data['direction']
                })
                
                if kalman_velocity_data['actual_velocity_mps'] is not None:
                    tracking_entry['velocity_mps'] = kalman_velocity_data['actual_velocity_mps']
            
            self.tracking_data.append(tracking_entry)
            
            # Visualize raw detection if available
            if detected_bbox is not None:
                x1, y1, x2, y2 = map(int, detected_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red = raw detection
        
        return frame

    def save_tracking_data(self, output_path):
        """
        Save tracking data to CSV
        """
        df = pd.DataFrame(self.tracking_data)
        df.to_csv(output_path, index=False)
        return df

def extract_clip(input_path, output_path, start_sec=0, duration_sec=5):
    """
    Extract a clip from a video
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    start_frame = int(start_sec * fps)
    total_frames = int(duration_sec * fps)
    end_frame = start_frame + total_frames
    
    # Set position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Extract frames
    print(f"[INFO] Extracting {duration_sec} seconds from video starting at {start_sec} seconds...")
    
    frame_count = 0
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Write frame to output
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Print every 30 frames (about 1 second)
            progress = frame_count / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"[INFO] Extracted clip saved to: {output_path}")
    print(f"[INFO] Extracted {frame_count} frames ({frame_count/fps:.2f} seconds)")
    
    return {
        "output_path": output_path,
        "fps": fps,
        "duration": frame_count/fps,
        "frame_count": frame_count
    }

def process_video(input_path, output_path, model_path, data_output_path, 
                 conf_threshold=0.3, iou_threshold=0.5, drone_class_id=0,
                 start_frame=0, end_frame=None, max_height_meters=4.0):
    """
    Process video with drone tracking
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set end frame if not specified
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    # Initialize tracker
    tracker = DroneTracker(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        drone_class_id=drone_class_id,
        max_height_meters=max_height_meters,
        fps=fps
    )
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing video from frame {start_frame} to {end_frame}...")
    print(f"[INFO] Video FPS: {fps}, Resolution: {width}x{height}")
    print(f"[INFO] Max drone height set to {max_height_meters} meters")
    
    # Process frames
    for frame_idx in tqdm(range(start_frame, end_frame), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = tracker.process_frame(frame, frame_idx)
        
        # Write frame
        out.write(processed_frame)
    
    # Clean up
    cap.release()
    out.release()
    
    # Save tracking data
    tracker.save_tracking_data(data_output_path)
    
    print(f"[INFO] Smoothed output saved to: {output_path}")
    print(f"[INFO] Tracking data saved to: {data_output_path}")
    
    return tracker.tracking_data

def main():
    from types import SimpleNamespace

    # Parameters
    args = SimpleNamespace(
    input="/kaggle/input/specific-videos/frogJump.mp4",
    output="smoothed_ekf_drone_tracking_frogjump.mp4",
    data_output="drone_tracking_data_frogjump.csv",
    model="/kaggle/input/new-yolov8/yolov8_new.pt",
    extract_first=0,  # Change this to 0
    extract_start=14, # Add this new parameter for start time
    extract_duration=8, # Add this new parameter for duration (22-14=8)
    extract_output="seconds_14_to_22_frogjump.mp4", # Change the output filename if desired
    max_height=4.0,
    conf=0.3,
    iou=0.5,
    class_id=0
)

    # Extract clip if requested
    extracted_clip_info = None
    if args.extract_start >= 0 and args.extract_duration > 0:
        print(f"[INFO] Extracting {args.extract_duration} seconds starting from {args.extract_start} seconds to {args.extract_output}")
        extracted_clip_info = extract_clip(
            input_path=args.input,
            output_path=args.extract_output,
            start_sec=args.extract_start,
            duration_sec=args.extract_duration
        )
        
        # If we extracted a clip, process that instead
        if extracted_clip_info:
            input_to_process = args.extract_output
            print(f"[INFO] Processing extracted clip: {input_to_process}")
        else:
            input_to_process = args.input
    else:
        input_to_process = args.input
    
    # Process video with tracking
    tracking_data = process_video(
        input_path=input_to_process,
        output_path=args.output,
        model_path=args.model,
        data_output_path=args.data_output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        drone_class_id=args.class_id,
        max_height_meters=args.max_height
    )
    
    # Print summary
    if tracking_data:
        print(f"[INFO] Processed {len(tracking_data)} frames")
        print(f"[INFO] Average confidence: {np.mean([d['confidence'] for d in tracking_data]):.3f}")
        print(f"[INFO] Frames with detections: {sum([1 for d in tracking_data if d['detected']])}")
        
        # Calculate average velocity if available
        if 'velocity_mps' in tracking_data[0]:
            avg_velocity = np.mean([d['velocity_mps'] for d in tracking_data if 'velocity_mps' in d])
            print(f"[INFO] Average velocity: {avg_velocity:.2f} m/s")
            
            # Calculate max velocity
            max_velocity = np.max([d['velocity_mps'] for d in tracking_data if 'velocity_mps' in d])
            print(f"[INFO] Maximum velocity: {max_velocity:.2f} m/s")

if __name__ == "__main__":
    main()
    
    
    
    
    


