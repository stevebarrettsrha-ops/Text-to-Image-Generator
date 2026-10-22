#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Started from the file manager (right-click → Run as a Program) there is no
# terminal: nothing this script says can be seen, and a failure just looks like
# nothing happening. On a Linux desktop, open a terminal and run in it instead.
# Anything run with IDEOGRAM_STUDIO_NO_TERMINAL=1 (a service, an IDE task) stays
# where it was started. Not INVOCATION_ID: a file manager that runs as
# a systemd service hands it to everything it starts.
if [ "$(uname)" = "Linux" ] && [ -z "${IDEOGRAM_STUDIO_IN_TERMINAL:-}" ] \
   && [ ! -t 0 ] && [ ! -t 1 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] \
   && [ -z "${IDEOGRAM_STUDIO_NO_TERMINAL:-}" ]; then
  export IDEOGRAM_STUDIO_IN_TERMINAL=1
  SELF="$(pwd)/$(basename "$0")"
  for term in gnome-terminal ptyxis kgx konsole xfce4-terminal mate-terminal \
              tilix x-terminal-emulator xterm; do
    command -v "$term" >/dev/null 2>&1 || continue
    case "$term" in
      gnome-terminal|ptyxis|kgx) "$term" -- bash "$SELF" && exit 0 ;;
      xfce4-terminal|mate-terminal) "$term" -x bash "$SELF" && exit 0 ;;
      *) "$term" -e bash "$SELF" && exit 0 ;;
    esac
  done
  # No window opened: this run carries on here, so it is not one.
  unset IDEOGRAM_STUDIO_IN_TERMINAL
  # No terminal found: carry on without one. The page still opens.
fi

# In a window we opened, keep it open on failure so the reason can be read.
pause_on_error() {
  local code=$?
  if [ -n "${IDEOGRAM_STUDIO_IN_TERMINAL:-}" ] && [ "$code" -ne 0 ] \
     && [ "$code" -ne 130 ] && [ -t 0 ]; then
    printf "\n  Ideogram Studio stopped (exit code %s). Press Enter to close." "$code"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found. Install it, then run this again." >&2
  exit 1
fi
# The studio's own small venv (Flask, requests). Current Debian/Ubuntu refuse
# pip into the system Python (PEP 668). A venv without pip is rebuilt: without
# python3-venv, `venv` dies after writing bin/python but before pip, and that
# half-built folder must not be kept.
# A folder first set up on Windows (a drive both systems share) can carry a
# Windows .venv — Scripts/python.exe and no bin/python. It can never run
# here, and rebuilding over it would break it for Windows, so this system
# gets its own beside it.
VENV=.venv
if [ -e .venv/Scripts/python.exe ] && [ ! -e .venv/bin/python ]; then
  VENV=".venv-$(uname -s | tr '[:upper:]' '[:lower:]')"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$VENV"
  "$PY" -m venv "$VENV" || {
    rm -rf "$VENV"
    echo "Could not create a virtual environment. On Debian/Ubuntu install" >&2
    echo "python3-venv (sudo apt-get install -y python3-venv), then run this again." >&2
    exit 1
  }
fi
PY="$VENV/bin/python"
echo "  Using: $($PY -c 'import sys;print(sys.executable)')"
"$PY" -m pip install --disable-pip-version-check --quiet -r requirements.txt
# The server takes this script's place (exec), so whatever stops the
# script stops the server. Only in a window this script opened does
# the shell stay, so the trap above can keep a failure on screen.
if [ -n "${IDEOGRAM_STUDIO_IN_TERMINAL:-}" ]; then
  "$PY" server.py
else
  exec "$PY" server.py
fi
