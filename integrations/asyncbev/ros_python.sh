#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "$ROOT/ros2/install/setup.bash"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 "$@"
