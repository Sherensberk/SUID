from core import system_pb2_grpc
from core import system_pb2
from core.core import serve, logging
from codetiming import Timer

from interface.camera_interface import (
    Basler,
    OpenCV,
    cv2,
)

import numpy as np

logger = logging.getLogger("CameraService")
class CameraServicer(system_pb2_grpc.CameraServicer):
    def __init__(self, cam) -> None:
        super().__init__()
        self.cam = cam
    
    @Timer("take_picture",  text="{name} elapsed time: {:.4f}s", logger=logger.debug)
    def take_picture(self, request, context):
        status, frame = cam.read_raw()
        if status:
            return system_pb2.Image(
                width=frame.shape[1],
                height=frame.shape[0],
                data=cv2.imencode('.jpg', frame)[1].tobytes())
        return system_pb2.Image(
            width=0,
            height=0,
            data=cv2.imencode('.jpg', np.zeros((1,1,3), dtype="uint8"))[1].tobytes())

if __name__ == '__main__':
    import argparse
    from threading import Event

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="50051")
    parser.add_argument('--source', type=str, help="(SOURCE) ", required=False)
    parser.add_argument('--type', choices=["USB", "BASLER"], required=True)

    args = parser.parse_args()
    stop_signal = Event()

    match args.type:
        case 'USB':
            config = {'index': args.source or 0 , cv2.CAP_PROP_FRAME_WIDTH:1920, cv2.CAP_PROP_FRAME_HEIGHT:1080}
            cam = OpenCV(config, stop_signal=stop_signal)
        case 'BASLER':
            cam = Basler(configuration="cameraSettings", stop_signal=stop_signal)
    
    servicer = CameraServicer(cam)
    serve(system_pb2_grpc.add_CameraServicer_to_server, servicer, logger, stop_signal, server_port=args.service_port)
    cam.thread.join()