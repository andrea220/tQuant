"""``python -m tqstudio [--port 8000]``: start the web app on all network interfaces."""

import argparse
import socket

import uvicorn

parser = argparse.ArgumentParser(prog="tqstudio")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()
host = socket.gethostname()
print(f"tqstudio: http://localhost:{args.port} (this pc), http://{host}:{args.port} or "
      f"http://{socket.gethostbyname(host)}:{args.port} (colleagues on the same network)")
uvicorn.run("tqstudio.server:app", host="0.0.0.0", port=args.port)
