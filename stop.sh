#!/usr/bin/env bash
# Stop the AI Lab GUI: kill the process AND free its port.
#   ./stop.sh [port]      (default 8501, or set AILAB_PORT)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${AILAB_PORT:-${1:-8501}}"
PIDFILE="$HERE/.gui.pid"
stopped=0

# 1) Kill the process recorded in the pidfile (regardless of port).
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null && stopped=1
  fi
  rm -f "$PIDFILE"
fi

# 2) Catch any Streamlit running our app on any port.
if pkill -f "streamlit run gui/app.py" 2>/dev/null; then
  stopped=1
fi

# 3) Free the port - kill anything still listening on it.
PIDS="$(lsof -ti "tcp:$PORT" 2>/dev/null || true)"
if [ -n "$PIDS" ]; then
  echo "Freeing port $PORT (PIDs: $(echo "$PIDS" | tr '\n' ' '))..."
  kill $PIDS 2>/dev/null || true
  sleep 1
  PIDS="$(lsof -ti "tcp:$PORT" 2>/dev/null || true)"   # force-kill survivors
  if [ -n "$PIDS" ]; then
    kill -9 $PIDS 2>/dev/null || true
  fi
  stopped=1
fi

if [ "$stopped" -eq 1 ]; then
  echo "AI Lab GUI stopped; port $PORT free."
else
  echo "Nothing to stop (no pidfile, no matching process, port $PORT already free)."
fi
