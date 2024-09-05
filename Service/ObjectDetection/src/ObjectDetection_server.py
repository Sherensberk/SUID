from core import system_pb2_grpc
from core import system_pb2
from core.core import serve, logging
from codetiming import Timer

from interface.darknet import Net

import cv2
import numpy as np

logger = logging.getLogger("ObjectDetectionService")
class ObjectDetection(system_pb2_grpc.ObjectDetectionServicer):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net
        
    @Timer("make_inference",  text="{name} elapsed time: {:.4f}s", logger=logger.debug)
    def make_inference(self, request, context):
        print('gotcha')
        frame = cv2.imdecode(np.frombuffer(request.data, dtype = np.uint8), cv2.IMREAD_COLOR)
        return system_pb2.InferenceResult(**self.net.predict(frame))

if __name__ == '__main__':
    import argparse
    from threading import Event
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="50051")
    parser.add_argument('--cfg', type=str, required=False, help="(cfg) filepath.", default="model.cfg")
    parser.add_argument('--names', type=str, required=False, help="(names) filepath.", default="model.names")
    parser.add_argument('--weights', type=str, required=False, help="(weights) filepath.", default="model_best.weights")
    
    args = parser.parse_args()
    stop_signal = Event()
    
    net =  Net((args.cfg).encode("utf-8"), (args.names).encode("utf-8"), (args.weights).encode("utf-8"))

    servicer = ObjectDetection(net)
    serve(system_pb2_grpc.add_ObjectDetectionServicer_to_server, servicer, logger, stop_signal, args.service_port)
