from __future__ import annotations

import argparse
import threading
import webbrowser

from ..web.server import PowerGridWebController, make_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PowerGrid web GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    parser.add_argument("--open", action="store_true", help="Open the GUI in the default browser.")
    parser.add_argument("--smoke-test", action="store_true", help="Validate web resources and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = PowerGridWebController()
    if args.smoke_test:
        meta = controller.metadata()
        if not meta["maps"] or not meta["controllers"]:
            raise RuntimeError("web GUI metadata is incomplete")
        print("PowerGrid web GUI smoke test passed.")
        return 0

    server = make_server(args.host, args.port, controller=controller)
    display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    url = f"http://{display_host}:{server.server_port}"
    print(f"PowerGrid Web GUI running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PowerGrid Web GUI.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
