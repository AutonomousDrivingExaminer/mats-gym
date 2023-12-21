#!/bin/bash
idx=${1:-0}
gpu=${2:-0}
CARLA_VERSION=${3:-0.9.15}
pfrom=$((2000 + $idx * 3))
pto=$((2002 + $idx * 3))
docker run --gpus "device="${gpu} \
          --name carla-${idx} \
           -p ${pfrom}-${pto}:${pfrom}-${pto} \
          --rm \
	       -it \
           -d \
          --privileged \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $(pwd)/reports/replays:/home/carla/replays \
           carlasim/carla:$CARLA_VERSION ./CarlaUE4.sh -carla-server -RenderOffScreen -world-port=${pfrom}