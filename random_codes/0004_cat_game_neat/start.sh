#!/usr/bin/env bash
# Cat Game NEAT - local static-server launcher (POSIX)
# Serves ONLY this project directory, bound EXPLICITLY to 127.0.0.1,
# on an ephemeral port generated locally. Direct file:// opening is
# unsupported by browsers, so use this launcher to play the game.
set -u

# (a) Serve only the script's own directory (the project root).
cd "$(dirname "$0")" || { echo "Cannot cd to script directory" >&2; exit 1; }

PY="python3"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 is required but was not found." >&2
  exit 1
fi

# (c) Locally generate a validated integer port via socket bind-to-port-0.
#     The -c program is a fixed literal; no env/CLI strings are evaluated.
PORT="$("$PY" -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
finally:
    s.close()
assert isinstance(p, int) and 1 <= p <= 65535, "bad port"
print(p)
')" || { echo "Failed to choose a local port." >&2; exit 1; }

# Validate the generated value is a pure decimal integer before any use.
case "$PORT" in
  ''|*[!0-9]*) echo "Invalid generated port: $PORT" >&2; exit 1 ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "Generated port out of range: $PORT" >&2
  exit 1
fi

# (b)/(c) The port appears ONLY in discrete command arguments and in this
#         single fixed URL literal. Bind loopback EXPLICITLY.
URL="http://127.0.0.1:${PORT}/index.html"

echo "Starting local static server at $URL"

# Run the server in the background so reachability checks can run first,
# then keep it in the foreground (wait) until Ctrl-C.
"$PY" -m http.server "$PORT" --bind 127.0.0.1 --directory . &
SERVER_PID=$!

cleanup() {
  trap - INT TERM EXIT
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping server (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# (d) Security/startup checks. The fixed Python helper below:
#     * waits for the server to answer on loopback,
#     * finds a local non-loopback IPv4 address, and
#     * confirms the server is NOT reachable on that address.
#     Exit codes: 0 = pass, 1 = not ready yet (retry), 2 = exposed externally.
CHECK_PY='
import socket, sys, time
port = int(sys.argv[1])

def probe(ip):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((ip, port))
        return True
    except OSError:
        return False
    finally:
        s.close()

deadline = time.monotonic() + 5.0
while not probe("127.0.0.1"):
    if time.monotonic() > deadline:
        sys.exit(1)
    time.sleep(0.2)
print("CHECK OK: reachable on loopback 127.0.0.1:" + str(port))

ips = set()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("192.0.2.1", 9))  # route selection only; no packets sent
    ips.add(s.getsockname()[0])
except OSError:
    pass
finally:
    s.close()
try:
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ips.add(info[4][0])
except OSError:
    pass
nonloop = sorted(ip for ip in ips if ip and not ip.startswith("127.") and ip != "0.0.0.0")
if not nonloop:
    print("CHECK WARN: no non-loopback interface to probe; --bind 127.0.0.1 still enforced")
    sys.exit(0)
ip = nonloop[0]
if probe(ip):
    print("CHECK FAIL: server IS reachable on non-loopback " + ip + ":" + str(port))
    sys.exit(2)
print("CHECK OK: NOT reachable on non-loopback " + ip + ":" + str(port) + " (loopback-only)")
sys.exit(0)
'

check_ok=0
for attempt in 1 2 3 4 5; do
  if "$PY" -c "$CHECK_PY" "$PORT"; then
    check_ok=1
    break
  else
    rc=$?
    if [ "$rc" -eq 2 ]; then
      echo "Refusing to continue: server is exposed on a non-loopback interface." >&2
      cleanup
      exit 1
    fi
    sleep 0.3
  fi
done
if [ "$check_ok" -ne 1 ]; then
  echo "Security/startup checks failed after retries." >&2
  cleanup
  exit 1
fi

# Open the fixed URL in the default browser (best effort).
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 &
elif command -v gio >/dev/null 2>&1; then
  gio open "$URL" >/dev/null 2>&1 &
elif "$PY" -c 'import webbrowser' 2>/dev/null; then
  "$PY" -c 'import webbrowser, sys; webbrowser.open(sys.argv[1])' "$URL" >/dev/null 2>&1 &
fi

echo
echo "============================================================"
echo " Cat Game NEAT is served at: $URL"
echo " Opening it in your browser now."
echo " This server listens on 127.0.0.1 ONLY - not on your network."
echo " Press Ctrl-C to stop the server."
echo "============================================================"
echo

# Keep the server running in the foreground until Ctrl-C.
wait "$SERVER_PID"
