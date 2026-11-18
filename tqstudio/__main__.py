"""``python -m tqstudio [--port 8010] [--data live|synthetic]``: start the web app on all network interfaces, on the
real snapshots (``marketdata/live``, default) or the synthetic ones (``marketdata``)."""

import argparse
import os
import socket

import uvicorn

parser = argparse.ArgumentParser(prog="tqstudio")
parser.add_argument("--port", type=int, default=8010)
parser.add_argument("--data", choices=["live", "synthetic"], default="live")
parser.add_argument("--marketdata", help="market data folder (default: the first valid of tqstudio/config.yaml)")
args = parser.parse_args()
os.environ["TQSTUDIO_DATA"] = args.data              # read by tqstudio.server at import
if args.marketdata:
    os.environ["TQSTUDIO_MARKETDATA"] = args.marketdata   # also inherited by the download / calibration processes
from tqstudio.paths import MARKETDATA  # noqa: E402  (resolved now: the override above, or the config)

os.environ["TQSTUDIO_MARKETDATA"] = str(MARKETDATA)
host = socket.gethostname()
print(f"tqstudio: http://localhost:{args.port} (this pc), http://{host}:{args.port} or "
      f"http://{socket.gethostbyname(host)}:{args.port} (colleagues on the same network); data: {args.data}; "
      f"market data folder: {MARKETDATA}")
uvicorn.run("tqstudio.server:app", host="0.0.0.0", port=args.port)
