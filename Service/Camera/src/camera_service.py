from threading import Thread, Event

from interface.camera_interface import (
    Basler,
    OpenCV,
    WorkingArea,
    cv2,
)
import numpy as np
from core.core import service
from codetiming import Timer

@Timer(name="CaptureImageProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, cam):
    action = command.get('action')
    match action:
        case 'take_picture':
            if isinstance(cam, WorkingArea):
                (status, frame, vecs) = cam.read_raw() 
            else:
                status, frame = cam.read_raw()
                vecs = cam.vecs
            if status:
                cv2.imwrite('/src/raw.jpg', frame)
            return {'action':'image', 'status':status, 'value':{'original_image':frame}, 'info':{'vecs': vecs, 'matrix': cam.matrix, 'from':'camera'},'to_output':True}
        case _:
            return {'status':False, 'frame':None, 'to_output':False}

stop_signal = Event()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="4000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="filter")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="5000")
    parser.add_argument('--source', type=str, help="(SOURCE) ", required=False)
    parser.add_argument('--type', choices=["USB", "BASLER"], required=True)

    args = parser.parse_args()
    print("Source:", args.source)
    match args.type:
        case 'USB':
            config = {'index': args.source or 0 , cv2.CAP_PROP_FRAME_WIDTH:1920, cv2.CAP_PROP_FRAME_HEIGHT:1080}
            cam = OpenCV(config, stop_signal=stop_signal)
        case 'BASLER':
            cam = Basler(configuration="cameraSettings", stop_signal=stop_signal)

    callback_args = (cam, )
    kwargs={'stop_signal':stop_signal, }
    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args), kwargs=kwargs, daemon=True)

    print(f"{__file__} started.")
    server_process.start()
    cam.thread.join()
    if stop_signal.is_set():
        exit(1)
    server_process.join()
    print(f"{__file__} stopped.")
