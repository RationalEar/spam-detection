#!/bin/bash
set -e

if [ "$1" = 'jupyter' ]; then
  echo "Starting Jupyter Lab..."
  # Use --ip=0.0.0.0 to make it accessible from outside the container
  # --no-browser prevents opening a browser inside the container
  # --allow-root is often needed for Docker containers (use with caution in prod)
  # --port=8888 is default, but good to be explicit
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
elif [ "$1" = 'script' ]; then
  shift 1 # Remove 'script' from arguments
  echo "Running script: $@"
  python "$@"
else
  # Default behavior if no specific command matches, or pass through
  exec "$@"
fi