#!/bin/bash
CARLA_VERSION=${2:-0.9.15}
pfrom=${1:-2000}
pto=$(($pfrom + 3))
echo "Starting carla server on ports ${pfrom}-${pto} with version ${CARLA_VERSION}"
docker run --gpus all \
          --name carla-${pfrom} \
          --rm \
	       -it \
          --privileged \
           -p ${pfrom}-${pto}:${pfrom}-${pto} \
           -e DISPLAY=$DISPLAY \
           -e SDL_VIDEODRIVER=x11 \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $(pwd)/reports/replays:/home/carla/replays \
           adex/carla:$CARLA_VERSION ./CarlaUE4.sh -carla-server -world-port=${pfrom}
