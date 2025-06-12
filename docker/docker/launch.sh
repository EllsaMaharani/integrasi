#!/bin/bash

# MLFF System Launcher
echo "===================================================="
echo "🚀 MLFF System Launcher"
echo "===================================================="

# Determine if running on Jetson by checking for CUDA
if [ -d "/usr/local/cuda" ]; then
    DEVICE="cuda:0"
    echo "🖥️ Running on Jetson with CUDA"
else
    DEVICE="cpu"
    echo "🖥️ Running on CPU only"
fi

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Start roscore if not already running
if ! pgrep -x roscore > /dev/null; then
    echo "Starting ROS core..."
    roscore &
    # Wait for roscore to start
    sleep 3
else
    echo "ROS core is already running"
fi

# Buat logs directory
mkdir -p logs

# Copy model weight ke direktori Effdet
mkdir -p /home/effdet
cp /home/ros_ws/inference_graph/saved_model /home/effdet/

# Set PYTHONPATH to include object_detection module
export PYTHONPATH="/tensorflow/models/research:/tensorflow/models/research/slim:$PYTHONPATH"

# Determine source (camera or video)
if [ -e "/dev/video0" ] && [ "$DEVICE" = "cuda:0" ]; then
    SOURCE="0"
    echo "📹 Using camera as source"
else
    SOURCE="/home/ros_ws/waskita_ent1.mp4"
    echo "🎞️ Using video file"
fi

# Start EfficientDet detector
echo "1. Starting EfficientDet detector..."
cd /home/ros_ws 
echo "   Current directory: $(pwd)"
echo "   Checking if bismillah_pengujian.py exists: $(ls -la bismillah_pengujian.py 2>/dev/null || echo 'NOT FOUND')"
echo "   Python version: $(python3 --version)"
echo "   TensorFlow version: $(python3 -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'TensorFlow not available')"
python3 bismillah_pengujian.py > /home/ros_ws/logs/detector.log 2>&1 &
DETECTOR_PID=$!
echo "   EFFICIENT DET DETECTOR started (PID: $DETECTOR_PID)"

# Kembali ke direktori utama
cd /home/ros_ws

# Start sensor reader
echo "2. Starting sensor reader..."
python3 sensor.py > logs/sensor.log 2>&1 &
SENSOR_PID=$!
echo "   Sensor reader started (PID: $SENSOR_PID)"

# Start subscriber with Firebase integration
echo "3. Starting MLFF subscriber with Firebase..."
python3 subscriber.py > logs/subscriber.log 2>&1 &
SUBSCRIBER_PID=$!
echo "   MLFF subscriber started (PID: $SUBSCRIBER_PID)"

echo "===================================================="
echo "✅ All MLFF components started successfully!"
echo "Logs are being saved to logs/ directory"
echo "Press Ctrl+C to stop all components"
echo "===================================================="

# Wait a moment for logs to be created
sleep 2

# Show logs in real-time (with error handling)
find logs -type f -name "*.log" | xargs tail -f || echo "No log files found yet"

# Wait for all processes
wait

# Fungsi untuk membersihkan proses saat keluar
cleanup() {
    echo ""
    echo "🛑 Stopping all processes..."
    
    # Hentikan proses berdasarkan PID
    for pid in $SUBSCRIBER_PID $SENSOR_PID $DETECTOR_PID; do
        if ps -p $pid > /dev/null; then
            echo "   Stopping process PID: $pid"
            kill -SIGINT $pid 2>/dev/null || kill -9 $pid 2>/dev/null
        fi
    done
    
    # Hentikan roscore jika kita memulainya
    if [ -n "$ROSCORE_PID" ]; then
        echo "   Stopping roscore"
        kill -SIGINT $ROSCORE_PID 2>/dev/null || kill -9 $ROSCORE_PID 2>/dev/null
    fi
    
    echo "✅ All processes stopped"
    exit 0
}

# Register trap
trap cleanup SIGINT SIGTERM

# Wait for Ctrl+C
echo "Press Ctrl+C to stop all components"
wait