#!/bin/bash
# Start the peak3d app (macOS and Linux): runs the peak3d Docker image and opens it in your browser.
# Needs Docker Desktop (macOS) or Docker Engine (Linux). On macOS, double-click "Start peak3d.command".
#
# The app sees your home folder (same paths as on this computer). Optional settings:
#   PEAK3D_DATA=/path/to/folder   a different folder for the app to see
#   PEAK3D_IMAGE=...              a different image or version
#   PEAK3D_NO_WAIT=1              start, print the link and return (the app keeps running)

IMAGE="${PEAK3D_IMAGE:-ghcr.io/fanzhoukong/peak3d:gui-prototype}"
DATA="${PEAK3D_DATA:-$HOME}"
NAME=peak3d-gui
MAC=0; [ "$(uname)" = Darwin ] && MAC=1

say() { printf '%s\n' "$*"; }
open_url() {
  if [ $MAC = 1 ]; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
  fi
}
fail() {
  say ""
  say "$*"
  [ $MAC = 1 ] && [ -z "$PEAK3D_NO_WAIT" ] && read -r -p "Press Return to close this window. " _
  exit 1
}
url_of() {   # the link of the running app: its port and token
  local port token
  port=$(docker port "$NAME" 8765/tcp 2>/dev/null | head -n 1 | sed 's/.*://')
  token=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME" | sed -n 's/^PEAK3D_TOKEN=//p')
  [ -n "$port" ] && [ -n "$token" ] && printf 'http://127.0.0.1:%s/?token=%s' "$port" "$token"
}

say "peak3d"
say "------"

# 1. Docker installed and running
if ! command -v docker >/dev/null 2>&1; then
  open_url "https://www.docker.com/products/docker-desktop/"
  fail "Docker is not installed. Install Docker Desktop (the download page just opened), start it once," \
       "then start peak3d again."
fi
if ! docker info >/dev/null 2>&1; then
  say "Starting Docker..."
  [ $MAC = 1 ] && open -a Docker >/dev/null 2>&1
  for _ in $(seq 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker Desktop, wait until it says it is running, and try again."
fi

# 2. Already running? Just open it again (does not interrupt a run in progress)
if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; then
  URL=$(url_of)
  if [ -n "$URL" ]; then
    open_url "$URL"
    say "peak3d is already running: $URL"
    exit 0
  fi
  docker rm -f "$NAME" >/dev/null 2>&1
fi

# 3. Latest image (about 300 MB the first time; later only what changed)
say "Checking for the latest peak3d (the first time downloads about 300 MB)..."
if ! docker pull "$IMAGE"; then
  docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "Could not download $IMAGE. Check your internet connection and try again."
  say "Could not check for updates; using the copy on this computer."
fi

# 4. Start: your folder at the same path inside, a random free port on this computer only, a fresh token
[ -d "$DATA" ] || fail "Folder not found: $DATA"
DATA=$(cd "$DATA" && pwd -P)
TOKEN=$(od -An -N12 -tx1 /dev/urandom | tr -d ' \n')
docker rm -f "$NAME" >/dev/null 2>&1
USER_OPT=""
[ "$(uname)" = Linux ] && USER_OPT="--user $(id -u):$(id -g)"      # files you create stay yours
# shellcheck disable=SC2086
docker run -d --rm --init --name "$NAME" $USER_OPT -p 127.0.0.1::8765 -e PEAK3D_TOKEN="$TOKEN" -e HOME=/tmp \
  -v peak3d-cache:/var/cache/peak3d -v "$DATA":"$DATA" "$IMAGE" \
  gui --host 0.0.0.0 --port 8765 --root "$DATA" --no-browser >/dev/null \
  || fail "Could not start peak3d (see the message above)."

URL=$(url_of)
[ -n "$URL" ] || fail "peak3d started but its address could not be read; run: docker logs $NAME"
for _ in $(seq 60); do
  curl -fs -o /dev/null -H "X-Peak3d-Token: $TOKEN" "${URL%%/\?*}/api/info" && break
  sleep 1
done
open_url "$URL"
say ""
say "peak3d is running at:"
say "    $URL"
say "It can see the folder $DATA."
[ -n "$PEAK3D_NO_WAIT" ] && exit 0
say "Keep this window open while you use peak3d. Close it, or press Ctrl+C, to stop peak3d."

# 5. Stop the app when this window closes
trap 'say "Stopping peak3d..."; docker stop "$NAME" >/dev/null 2>&1; exit 0' INT TERM HUP
docker wait "$NAME" >/dev/null 2>&1 &
wait $!                      # (wait, unlike a foreground command, is interrupted by the trap at once)
