from core import system_pb2_grpc
from core import system_pb2
from core.core import serve, logging
from codetiming import Timer

import cv2
import numpy as np

from interface.worldmapping import Item, Plane

logger = logging.getLogger("ObjectPositioningService")
class ObjectPositioning(system_pb2_grpc.ObjectPositioningServicer):
    def __init__(self, plane):
        self.plane  = plane
    
    @Timer("make_worldmapping",  text="{name} elapsed time: {:.4f}s", logger=logger.debug)
    def make_worldmapping(self, request, context):
        print('gotcha')
        frame = cv2.imdecode(np.frombuffer(request.image.data, dtype = np.uint8), cv2.IMREAD_COLOR)
        self.plane.update_real_image(frame)
        centers = []
        for file in request.inference.file:
            for prediction in file.prediction:
                    centers.append(
                         Item(
                            f"({prediction.prediction_index}) {prediction.name}",
                            {
                                'x':prediction.original_point.x*file.original_width,
                                'y':prediction.original_point.y*file.original_height,
                            },
                            plane=self.plane
                        ).coordinate.real
                    )
        return system_pb2.WorldMappingResult(coordinates=centers)

if __name__ == '__main__':
    import argparse
    import cv2.aruco as aruco
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="50051")

    parser.add_argument('--working_plane', action='store_true')
    parser.add_argument('--marker_type', type=str, required=False )
    parser.add_argument('--board_shape_width', type=str, required=False)
    parser.add_argument('--board_shape_height',type=str, required=False)
    parser.add_argument('--square_size', type=str, required=False)
    parser.add_argument('--marker_size', type=str, required=False) 
    parser.add_argument('--id_offset', type=str, required=False)

    args = parser.parse_args()
    plane = Plane(
        getattr(aruco, args.marker_type),
        (int(args.board_shape_width), int(args.board_shape_height)),
        float(args.square_size),
        float(args.marker_size),
        int(args.id_offset)
    )
    RI, URI, DI, ROI = plane.calibrate_from_dir('./calibration/charuco/')

    servicer = ObjectPositioning(plane)
    serve(system_pb2_grpc.add_ObjectPositioningServicer_to_server, servicer, logger, server_port=args.service_port)