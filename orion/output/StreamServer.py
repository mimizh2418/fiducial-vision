import logging
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from ..config import Config
from ..pipeline import CaptureFrame

logger = logging.getLogger(__name__)

IMAGE_DOWNSCALE_FACTOR = 0.50


class StreamServer:
    _config: Config

    _frame: cv2.Mat
    _resolution_width: int
    _resolution_height: int
    _has_frame: bool = False

    def __init__(self, config: Config):
        self._config = config

    def _make_handler(self_mjpeg):  # type: ignore
        class StreamingHandler(BaseHTTPRequestHandler):
            HTML = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Orion Debug</title>
            <style>
                body {{
                    background-color: black;
                }}

                img {{
                    position: absolute;
                    left: 50%;
                    top: 50%;
                    transform: translate(-50%, -50%) scale({1.0 / IMAGE_DOWNSCALE_FACTOR});
                    max-width: 100%;
                    max-height: 100%;
                }}
            </style>
        </head>
        <body>
            <img src="stream.mjpg" />
        </body>
    </html>
            """

            def do_GET(self):
                if self.path == "/":
                    content = self.HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
                    )
                    self.end_headers()
                    try:
                        while True:
                            if not self_mjpeg._has_frame:
                                time.sleep(0.1)
                            else:
                                resized_frame = cv2.resize(self_mjpeg._frame,
                                                           None,
                                                           fx=IMAGE_DOWNSCALE_FACTOR,
                                                           fy=IMAGE_DOWNSCALE_FACTOR)
                                _, enc = cv2.imencode(".jpg", resized_frame)

                                frame_data = np.array(enc).tobytes()

                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(frame_data)))
                                self.end_headers()
                                self.wfile.write(frame_data)
                                self.wfile.write(b"\r\n")
                    except Exception as e:
                        logger.info(f"Removed streaming client {self.client_address}: {str(e)}")
                else:
                    self.send_error(404)
                    self.end_headers()

        return StreamingHandler

    class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    def _run(self, port: int) -> None:
        server = self.StreamingServer(("", port), self._make_handler())
        server.serve_forever()

    def start(self) -> None:
        logger.info("Starting stream server")
        threading.Thread(
            target=self._run, daemon=True, args=(self._config.network.stream_port,)
        ).start()

    def set_frame(self, frame: CaptureFrame) -> None:
        self._frame = frame.image
        self._resolution_height = frame.resolution_height
        self._resolution_width = frame.resolution_width
        self._has_frame = True
