"""``python -m tqstudio [--port 8010] [--data live|synthetic]``: start the web app on all network interfaces, on the
real snapshots (``marketdata/live``, default) or the synthetic ones (``marketdata``)."""

import argparse
import os
import socket

import uvicorn

parser = argparse.ArgumentParser(prog="tqstudio")
parser.add_argument("--port", type=int, default=8010)
parser.add_argument("--data", choices=["live", "synthetic"], default="live")
args = parser.parse_args()
os.environ["TQSTUDIO_DATA"] = args.data              # read by tqstudio.server at import
host = socket.gethostname()
print(f"tqstudio: http://localhost:{args.port} (this pc), http://{host}:{args.port} or "
      f"http://{socket.gethostbyname(host)}:{args.port} (colleagues on the same network); data: {args.data}")
uvicorn.run("tqstudio.server:app", host="0.0.0.0", port=args.port)
