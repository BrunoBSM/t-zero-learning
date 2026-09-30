FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /t-zero

RUN apt-get update && apt-get install --no-install-recommends ffmpeg libgl1 libegl1 libglfw3 libglew2.2 -y && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /work/requirements.txt
RUN pip install -r requirements.txt
# Box2D envs (LunarLander): box2d-py compiles against swig, which the isolated build env can't see
RUN pip install swig && pip install --no-build-isolation box2d-py==2.3.5 \
    && python -c "import gymnasium as gym; gym.make('LunarLander-v3')"
ENV MUJOCO_GL=egl

VOLUME ["/t-zero"]