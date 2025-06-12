import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import tensorflow.io as tf_io
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.framework import convert_to_constants
import cv2
import time
import os
import sys
import traceback
from datetime import datetime
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db

# Global variables for interactive editing
EDIT_MODE = False
CURRENT_ZONE_POINTS = []
SELECTED_ZONE_TYPE = None  # 'mask' or 'counting'
SELECTED_ZONE_INDEX = None

# Define counting and mask zones
# Format: [top-left, top-right, bottom-right, bottom-left]
COUNTING_ZONES = [
    # Lane 1 - Wider counting zone with better coverage
    [[798, 258], [804, 255], [900, 269], [899, 273]], #waskita1_sejajar belakang
    # Lane 2 - Adjusted to cover more area
    [[911, 276], [915, 273], [1018, 291], [1015, 294]] #waskita1_sejajar belakang
]

# Mask zones expanded to ensure better vehicle detection
MASK_ZONES = [
    # Lane 1 - Larger mask zone
    [[743, 276], [857, 230], [941, 239], [868, 294]], #waskita1_sejajar belakang
    # Lane 2 - Expanded coverage area
    [[947, 243], [1040, 254], [998, 316], [874, 298]] #waskita1_sejajar belakang
]

def mouse_callback(event, x, y, flags, param):
    global EDIT_MODE, CURRENT_ZONE_POINTS, SELECTED_ZONE_TYPE, SELECTED_ZONE_INDEX, COUNTING_ZONES, MASK_ZONES
    
    if not EDIT_MODE:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        CURRENT_ZONE_POINTS.append([x, y])
        print(f"Point added: [{x}, {y}]")
        
        # If we complete a polygon (4 points)
        if len(CURRENT_ZONE_POINTS) == 4:
            if SELECTED_ZONE_TYPE == 'mask':
                MASK_ZONES[SELECTED_ZONE_INDEX] = CURRENT_ZONE_POINTS.copy()
            elif SELECTED_ZONE_TYPE == 'counting':
                COUNTING_ZONES[SELECTED_ZONE_INDEX] = CURRENT_ZONE_POINTS.copy()
                
            CURRENT_ZONE_POINTS.clear()
            print(f"Zone {SELECTED_ZONE_TYPE} {SELECTED_ZONE_INDEX} updated!")
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if CURRENT_ZONE_POINTS:
            CURRENT_ZONE_POINTS.pop()
            print("Removed last point")

def draw_editing_overlay(frame):
    if not EDIT_MODE:
        return frame
    
    overlay = frame.copy()
    
    # Draw current editing points
    if CURRENT_ZONE_POINTS:
        # Draw points
        for point in CURRENT_ZONE_POINTS:
            cv2.circle(overlay, (point[0], point[1]), 5, (0, 0, 255), -1)
        
        # Draw lines between points
        for i in range(len(CURRENT_ZONE_POINTS)):
            if i > 0:
                cv2.line(overlay, 
                         tuple(CURRENT_ZONE_POINTS[i-1]), 
                         tuple(CURRENT_ZONE_POINTS[i]), 
                         (0, 0, 255), 2)
    
    # Show edit mode status
    status_text = f"Edit Mode: {SELECTED_ZONE_TYPE} Zone {SELECTED_ZONE_INDEX if SELECTED_ZONE_TYPE else ''}"
    cv2.putText(overlay, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show instructions
    instructions = [
        "Controls:",
        "E - Toggle Edit Mode",
        "M - Edit Mask Zone",
        "C - Edit Counting Zone",
        "Left Click - Add Point",
        "Right Click - Remove Last Point",
        "Q - Quit"
    ]
    
    y = 60
    for instruction in instructions:
        cv2.putText(overlay, instruction, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
    
    return overlay

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Video settings
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
VIDEO_FPS = 60

# Output settings
OUTPUT_DIR = "output_videos"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class ZoneNormalizer:
    def __init__(self, reference_image_path):
        """Initialize with the reference image (the one used to create coordinates)"""
        ref_img = cv2.imread(reference_image_path)
        self.ref_height, self.ref_width = ref_img.shape[:2]
        
    def normalize_coordinates(self, coordinates):
        """Convert absolute coordinates to relative (0-1) coordinates"""
        return [
            [x / self.ref_width, y / self.ref_height]
            for x, y in coordinates
        ]
    
    def denormalize_coordinates(self, normalized_coords, target_width, target_height):
        """Convert relative coordinates back to absolute for the target image size"""
        return [
            [int(x * target_width), int(y * target_height)]
            for x, y in normalized_coords
        ]
    
def load_model(model_path):
    try:
        print(f"Attempting to load model from: {os.path.abspath(model_path)}")
        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            return None
            
        # Load saved model
        print("Loading saved model...")
        detect_fn = tf.saved_model.load(model_path)
        print("Model loaded successfully")
        return detect_fn
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None
    # try:
    #     with tf_io.gfile.GFile(path, 'r') as fid:
    #         label_map_string = fid.read()
    #         return label_map_string
    # except Exception as e:
    #     print(f"Error loading label map: {e}")
    #     return None

def display_model_summary(detect_fn):
    print("\nGenerating model summary...")
    # Extract the concrete function
    concrete_func = detect_fn.signatures["serving_default"]
    
    # Convert the model to a frozen graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    
    # Count the number of layers (operations in the graph)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print(f"Number of layers: {len(layers)}")
    
    # Count parameters from the frozen function graph
    total_parameters = 0
    for operation in frozen_func.graph.get_operations():
        if operation.type.lower() in ['const']:
            try:
                tensor = operation.outputs[0]
                shape = tensor.shape
                if shape.rank is not None:  # Check if shape is known
                    params = 1
                    for dim in shape:
                        if dim is not None:  # Check if dimension is known
                            params *= dim
                    total_parameters += params
            except:
                continue
    
    print(f"Number of parameters: {total_parameters:,}")
    print("Model summary complete.")

# UI Constants
margin = 10
header_height = 40
line_height = 30

def point_in_polygon(point, polygon):
    """
    Menggunakan algoritma Point in Polygon (PIP) dengan metode Winding Number
    Args:
        point: tuple (x, y) koordinat titik yang dicek
        polygon: list of tuples [(x1,y1), (x2,y2), ..] koordinat vertex polygon
    Returns:
        bool: True jika point di dalam polygon, False jika di luar
    """
    x, y = point
    wn = 0  # Winding number

    # Loop through polygon edges
    for i in range(len(polygon)):
        # Get vertex coordinates
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        # Check if point is above minimum y coordinate
        if y1 <= y:
            # Check if point is below maximum y coordinate
            if y2 > y:
                # Check if point is to the left of edge
                if (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                    wn += 1
        else:
            if y2 <= y:
                # Check if point is to the left of edge
                if (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
                    wn -= 1

    return wn != 0
    # x, y = point
    # n = len(polygon)
    # inside = False
    # p1x, p1y = polygon[0]
    # for i in range(n + 1):
    #     p2x, p2y = polygon[i % n]
    #     if y > min(p1y, p2y):
    #         if y <= max(p1y, p2y):
    #             if x <= max(p1x, p2x):
    #                 if p1y != p2y:
    #                     xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
    #                     if p1x == p2x or x <= xinters:
    #                         inside = not inside
    #     p1x, p1y = p2x, p2y
    # return inside
# def box_polygon_intersection(box, polygon):
#     """
#     Menghitung persentase box yang beririsan dengan polygon
#     box: [x1, y1, x2, y2] - koordinat box
#     polygon: list of [x, y] - koordinat polygon
#     """
#     # Buat mask untuk box dan polygon
#     x1, y1, x2, y2 = box
#     mask_box = np.zeros((max(y2+10, 1000), max(x2+10, 1000)), dtype=np.uint8)
#     cv2.rectangle(mask_box, (x1, y1), (x2, y2), 255, -1)
    
#     # Buat mask untuk polygon
#     mask_polygon = np.zeros_like(mask_box)
#     points = np.array(polygon, dtype=np.int32)
#     cv2.fillPoly(mask_polygon, [points], 255)
    
#     # Hitung intersection dan union
#     intersection = cv2.bitwise_and(mask_box, mask_polygon)
#     box_area = (x2 - x1) * (y2 - y1)
#     intersection_area = cv2.countNonZero(intersection)
    
#     # Hitung persentase irisan terhadap box
#     percentage = (intersection_area / box_area) * 100
#     return percentage


def detect_objects(bgr_frame, detect_fn, category_index, vehicle_counters, counting_zones, mask_zones):
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        height, width = rgb_frame.shape[:2]
        
        # Define color mapping for specific classes
        color_map = {
            1: (0, 0, 255),      # Red for G1 (BGR format)
            2: (203, 192, 255),  # Pink for G2 (BGR format)
            3: (144, 238, 144),  # Green for G3 (BGR format)
            4: (0, 165, 255),    # Orange for G4 (BGR format)
            5: (255, 255, 0)     # Cyan for G5 (BGR format)
        }
        
        # Convert RGB frame to tensor
        input_tensor = tf.convert_to_tensor(np.expand_dims(rgb_frame, 0), dtype=tf.uint8)
        
        # Start time for inference
        start_time = time.time()
        
        # Perform detection
        detections = detect_fn(input_tensor)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Convert detections to numpy arrays
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()
        
        # Dictionary to store object counts
        object_counts = {}
        
        # Only keep detections with score > 0.25 and apply basic NMS
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=100,
            iou_threshold=0.5, score_threshold=0.25
        ).numpy()
        
        # Draw mask zones first in green (they appear under everything)
        for zone in mask_zones:
            # Convert zone points to numpy array
            zone_np = np.array(zone, dtype=np.int32)
            # Draw filled polygon with transparency
            overlay = bgr_frame.copy()
            cv2.fillPoly(overlay, [zone_np], (0, 255, 0))  # Green color
            cv2.addWeighted(overlay, 0.2, bgr_frame, 0.8, 0, bgr_frame)
            # Draw zone border
            cv2.polylines(bgr_frame, [zone_np], True, (0, 255, 0), 2)

        # Draw counting zones next (so they appear under detections but over mask zones)
        for i, zone in enumerate(counting_zones):
            # Convert zone points to numpy array
            zone_np = np.array(zone, dtype=np.int32)
            # Draw filled polygon with transparency
            overlay = bgr_frame.copy()
            if i == 0:  # Lane 1 - Orange
                cv2.fillPoly(overlay, [zone_np], (0, 0, 255))
            else:  # Lane 2 - Cyan
                cv2.fillPoly(overlay, [zone_np], (0, 0, 255))
            cv2.addWeighted(overlay, 0.3, bgr_frame, 0.7, 0, bgr_frame)
            # Draw zone border
            cv2.polylines(bgr_frame, [zone_np], True, (255, 255, 255), 2)
            # Add zone label
            text_x = sum(p[0] for p in zone) // len(zone)
            text_y = sum(p[1] for p in zone) // len(zone)
            cv2.putText(bgr_frame, f"Zone {i+1}", (text_x-20, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Process and draw detections
        for i in selected_indices:
            # Get coordinates
            box = boxes[i]
            y1, x1 = int(box[0] * height), int(box[1] * width)
            y2, x2 = int(box[2] * height), int(box[3] * width)
            
            # Get class ID and color
            class_id = classes[i]
            color = color_map.get(class_id, (0, 0, 255))  # Default to red if class not in color_map
            
            # Draw bounding box
            cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 2)
            
            # Update object counts and check lane zones
            if class_id in category_index:
                class_name = category_index[class_id]['name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                
                # Calculate center point of detection
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check which lane the vehicle is in
                for lane_idx, zone in enumerate(counting_zones):
                    if point_in_polygon((center_x, center_y), zone):
                        vehicle_counters[lane_idx].increment(class_name)
                        break
                # Check which lane the vehicle is in
                # for lane_idx, zone in enumerate(counting_zones):
                #     # Hitung persentase irisan box dengan zona
                #     percent_in_zone = box_polygon_intersection([x1, y1, x2, y2], zone)
                #     # Jika lebih dari threshold (misal 20%), anggap kendaraan di zona
                #     if percent_in_zone > 20:  # Atur threshold sesuai kebutuhan
                #         vehicle_counters[lane_idx].increment(class_name)
                #         # Visualisasi persentase interseksi (opsional)
                #         cv2.putText(bgr_frame, f"{percent_in_zone:.1f}%", 
                #                     (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                #         break
                
                # Prepare label text
                label = f"{class_name}: {scores[i]:.2f}"
                
                # Draw label with background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(bgr_frame, 
                            (x1, y1 - label_height - baseline - 10),
                            (x1 + label_width + 10, y1),
                            color, 
                            cv2.FILLED)
                cv2.putText(bgr_frame, label,
                          (x1 + 5, y1 - baseline - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 2)
        # Format output string
        output_str = f"{width}x{height}"
        for obj_name, count in sorted(object_counts.items()):
            output_str += f" {count} {obj_name},"
        output_str = output_str.rstrip(',')
        output_str += f" Done. ({inference_time:.4f}s)"
        print(output_str)
        
        return bgr_frame, inference_time
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        traceback.print_exc()
        return bgr_frame, 0

# def load_labelmap(path):
#     """Load label map from file."""
#     try:
#         with tf_io.gfile.GFile(path, 'r') as fid:
#             label_map_string = fid.read()
#             return label_map_string
#     except Exception as e:
#         print(f"Error loading label map: {e}")
#         return None

def add_stats_overlay(frame, fps, vehicle_counters, inference_time):
    # Constants for layout
    margin = 20
    header_height = 40
    line_height = 25
    
    # Calculate total height needed
    total_height = header_height
    for i in range(2):
        total_height += line_height  # Lane header
        active_classes = sum(1 for count in vehicle_counters[i].class_counts.values() if count > 0)
        total_height += active_classes * line_height
        total_height += 10  # Spacing between lanes
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (300, margin + total_height),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw header
    cv2.putText(
        frame,
        "Vehicle Count Summary",
        (margin + 10, margin + 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    # Draw separator line
    cv2.line(
        frame,
        (margin, margin + header_height),
        (300, margin + header_height),
        (255, 255, 255),
        1
    )
    
    # Draw lane information
    y_offset = margin + header_height + 10
    for i, counter in enumerate(vehicle_counters):
        # Lane header
        cv2.putText(
            frame,
            f"Lane {i+1}: {counter.total_count}",
            (margin + 10, y_offset + line_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Vehicle class counts
        y_offset += line_height + 5
        for cls, count in counter.class_counts.items():
            if count > 0:  # Only show classes with vehicles
                cv2.putText(
                    frame,
                    f"{cls}: {count}",
                    (margin + 20, y_offset + line_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                y_offset += line_height
        
        y_offset += 10  # Add spacing between lanes
    
    # Add FPS and Inference Time at top right
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (frame.shape[1] - 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        f"Inference: {inference_time*1000:.1f}ms",
        (frame.shape[1] - 300, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    return frame

# Initialize Firebase
# cred = credentials.Certificate("D:\Deploy_Bitah\serviceAccountKey.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://mlff-e2d86-default-rtdb.firebaseio.com'
# })

# Reference to database root
# ref = db.reference('/')

# def send_to_firebase(vehicle_counters, timestamp, fps, inference_time):
#     try:
#         # Hitung total untuk setiap kelas kendaraan dari semua lane
#         total_by_class = {
#             'G1': 0,
#             'G2': 0,
#             'G3': 0,
#             'G4': 0,
#             'G5': 0
#         }
        
#         # Jumlahkan count dari semua lane
#         for counter in vehicle_counters:
#             for class_name, count in counter.class_counts.items():
#                 total_by_class[class_name] += count
        
#         # Data yang akan dikirim ke Firebase
#         data = {
#             'date': timestamp,
#             'g1': total_by_class['G1'],
#             'g2': total_by_class['G2'],
#             'g3': total_by_class['G3'],
#             'g4': total_by_class['G4'],
#             'g5': total_by_class['G5']
#         }
        
#         # Push data to Firebase
#         ref.child('vehicle_classification').push(data)
#         print("Data sent to Firebase successfully")
        
#     except Exception as e:
#         print(f"Error sending data to Firebase: {e}")

def main():
    # Declare global variables that will be modified
    global EDIT_MODE, CURRENT_ZONE_POINTS, SELECTED_ZONE_TYPE, SELECTED_ZONE_INDEX
    
    try:
        print("\nInitializing object detection with GPU acceleration...")
        print("TensorFlow version:", tf.__version__)
        print("Python version:", sys.version)
        
        print("\nLoading model... This might take a few minutes.")
        model_path = "/home/ros_ws/inference_graph/saved_model"
        label_map_path = "/home/ros_ws/objects_label_map.pbtxt"

        # Load model
        detect_fn = load_model(model_path)
        if detect_fn is None:
            print("Failed to load model")
            return

        # Langsung gunakan label_map_util:
        category_index = label_map_util.create_category_index_from_labelmap(
            label_map_path, use_display_name=True)
        
        if category_index is None:
            print("Failed to load label map")
            return
            
        # Display model summary
        display_model_summary(detect_fn)

        # Initialize vehicle counters list for each zone
        class VehicleCounter:
            def __init__(self):
                self.total_count = 0
                self.class_counts = {}
            
            def increment(self, vehicle_class):
                self.total_count += 1
                self.class_counts[vehicle_class] = self.class_counts.get(vehicle_class, 0) + 1

        # Create a counter for each counting zone
        vehicle_counters = [VehicleCounter() for _ in range(len(COUNTING_ZONES))]
        
        # Process video
        video_path = "/home/ros_ws/waskita_ent1.mp4"
        print(f"Processing video from: {os.path.abspath(video_path)}")
          # Open video capture
        # cap = cv2.VideoCapture(video_path)
        cap = cv2.VideoCapture(0)
        # Set resolusi yang diinginkan
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lebar 1920 pixels
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) # Tinggi 1080 pixels
        
        # Create window and set mouse callback
        cv2.namedWindow('Object Detection')
        cv2.setMouseCallback('Object Detection', mouse_callback)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
        
        # Prepare output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"Counting_baru_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_time = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_start = time.time()
            
            # Process frame (input is BGR frame from OpenCV)
            processed_frame, inference_time = detect_objects(frame, detect_fn, category_index, vehicle_counters, COUNTING_ZONES, MASK_ZONES)

            # Calculate FPS
            frame_time = time.time() - frame_start
            total_time += frame_time
            avg_fps = frame_count / total_time
            
            # Handle edit mode overlay
            if EDIT_MODE:
                processed_frame = draw_editing_overlay(processed_frame)
            
            # Add FPS and vehicle count overlay
            processed_frame = add_stats_overlay(processed_frame, avg_fps, vehicle_counters, inference_time)
            
            # Write and display frame
            out.write(processed_frame)
            cv2.imshow('Object Detection', processed_frame)
            
            # Handle keyboard input once per frame
            key = cv2.waitKey(1) & 0xFF
            
            # Handle edit mode first
            if key == ord('e'):  # Toggle edit mode
                EDIT_MODE = not EDIT_MODE
                if EDIT_MODE:
                    SELECTED_ZONE_TYPE = None
                    SELECTED_ZONE_INDEX = None
                    print("Edit mode ON - Press 'm' for mask zones or 'c' for counting zones")
                else:
                    CURRENT_ZONE_POINTS.clear()
                    SELECTED_ZONE_TYPE = None
                    SELECTED_ZONE_INDEX = None
                    print("Edit mode OFF")
            elif EDIT_MODE and key == ord('m'):  # Edit mask zone
                SELECTED_ZONE_TYPE = 'mask'
                if SELECTED_ZONE_INDEX is None or SELECTED_ZONE_TYPE != 'mask':
                    SELECTED_ZONE_INDEX = 0
                else:
                    SELECTED_ZONE_INDEX = (SELECTED_ZONE_INDEX + 1) % len(MASK_ZONES)
                CURRENT_ZONE_POINTS.clear()
                print(f"Editing mask zone {SELECTED_ZONE_INDEX + 1} of {len(MASK_ZONES)}")
            elif EDIT_MODE and key == ord('c'):  # Edit counting zone
                SELECTED_ZONE_TYPE = 'counting'
                if SELECTED_ZONE_INDEX is None or SELECTED_ZONE_TYPE != 'counting':
                    SELECTED_ZONE_INDEX = 0
                else:
                    SELECTED_ZONE_INDEX = (SELECTED_ZONE_INDEX + 1) % len(COUNTING_ZONES)
                CURRENT_ZONE_POINTS.clear()
                print(f"Editing counting zone {SELECTED_ZONE_INDEX + 1} of {len(COUNTING_ZONES)}")
            # Check for quit
            elif key == ord('q') or key == 27:  # 'q' key or ESC
                print("\nProcessing finished! Generating final statistics...")
                
                # Calculate final averages
                avg_inference = total_time/frame_count * 1000
                avg_fps = frame_count/total_time
                
                # Send final data to Firebase
                final_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_to_firebase(vehicle_counters, final_time, avg_fps, avg_inference)
                
                # Print final statistics with more detail
                print("\n=== Processing Statistics ===")
                print(f"Total frames processed: {frame_count}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Average inference time: {avg_inference:.2f}ms")
                print(f"Total processing time: {total_time:.2f} seconds")
                print("\n=== Vehicle Count Summary ===")
                for i, counter in enumerate(vehicle_counters):
                    print(f"\nLane {i+1} Summary:")
                    print(f"Total vehicles: {counter.total_count}")
                    if counter.total_count > 0:
                        for cls, count in counter.class_counts.items():
                            percentage = (count / counter.total_count) * 100
                            print(f"{cls}: {count} ({percentage:.1f}%)")
                
                break
            
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()

# Run the program
if __name__ == "__main__":
    main()
