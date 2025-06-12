FROM nvcr.io/nvidia/l4t-base:35.4.1

# Install ROS and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release curl gnupg2 \
    python3-pip python3-dev \
    git wget nano build-essential \
    libgl1-mesa-glx libglib2.0-0 libhdf5-dev libjpeg-dev \
    protobuf-compiler \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add ROS key and repo
RUN apt-get update && apt-get install -y lsb-release && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update

# Install ROS Noetic base
RUN apt-get update && apt-get install -y --fix-missing \
    ros-noetic-ros-base \
    ros-noetic-std-msgs \
    python3-rosdep && \
    if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    rosdep init && \
    mkdir -p /etc/ros/rosdep/sources.list.d && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install TensorFlow Jetson build (GPU support for JetPack 5.1.3)
RUN pip3 install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v511 tensorflow==2.11.0

# Install Python dependencies for EfficientDet, ROS and Computer Vision
RUN pip3 install --no-cache-dir \
    tf-slim tensorflow-hub tensorflow-addons \
    opencv-python numpy pandas matplotlib pillow \
    pyserial firebase-admin \
    rospkg catkin_pkg \
    protobuf==3.20.* \
    pycocotools Cython lxml absl-py

# Clone TensorFlow Models (EfficientDet inside)
RUN git clone --depth 1 https://github.com/tensorflow/models.git /tensorflow/models

# Compile protobufs and install Object Detection API
WORKDIR /tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .

# Set PYTHONPATH (safe way)
ENV PYTHONPATH="/tensorflow/models/research:/tensorflow/models/research/slim:${PYTHONPATH}"

# Back to workspace
WORKDIR /home/ros_ws

# Copy application files
COPY mlff-firebase-key.json ./
COPY bismillah_pengujian.py ./
COPY subscriber.py ./
COPY sensor.py ./
COPY launch.sh ./
COPY entrypoint.sh /entrypoint.sh
COPY waskita_ent1.mp4 ./waskita_ent1.mp4
COPY inference_graph/saved_model ./inference_graph/saved_model
COPY objects_label_map.pbtxt ./objects_label_map.pbtxt

# Make launch and entrypoint executable
RUN chmod +x /entrypoint.sh ./launch.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]