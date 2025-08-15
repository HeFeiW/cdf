# 使用 Ubuntu 22.04 基础镜像
FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    DISPLAY=:0 \
    QT_X11_NO_MITSHM=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    XAUTHORITY=/tmp/.docker.xauth

# 1. 首先安装curl和基础工具（必须先于任何curl操作）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 2. 配置国内镜像源（阿里云APT + 清华ROS）
RUN sed -i 's|http://.*archive.ubuntu.com|https://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list && \
    curl -sSL https://mirrors.tuna.tsinghua.edu.cn/rosdistro/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 3. 安装ROS Humble桌面版和其他依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    ros-dev-tools \
    x11-apps \
    dbus-x11 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 配置rosdep中科大源
RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    curl -o /etc/ros/rosdep/sources.list.d/20-default.list -L https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list && \
    echo 'export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml' >> ~/.bashrc && \
    # export ROSDISTRO_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml && \
    rosdep update
    
# # 创建非root用户
# RUN useradd -m rosuser && \
#     chown -R rosuser:rosuser /home/rosuser
# USER rosuser
# WORKDIR /home/rosuser

# rosdep update
# RUN rosdep update


# # 配置pip清华源
# RUN pip3 install --upgrade pip && \
#     pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 创建Xauth文件并设置工作目录
RUN touch /tmp/.docker.xauth && \
    chmod 777 /tmp/.docker.xauth
WORKDIR /workspace

# 初始化ROS环境
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

RUN apt-get update && \
    apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository universe && \
    rm -rf /var/lib/apt/lists/*
# 启动命令（示例）
CMD ["bash"]