import grpc
from core import system_pb2_grpc
from core import system_pb2

import cv2
import numpy as np
from time import sleep

def request_picture(stub: system_pb2_grpc.CameraStub):
    picture = stub.take_picture(system_pb2.NoneArgs())
    return picture
    return cv2.imdecode(np.frombuffer(picture.data, dtype = np.uint8), cv2.IMREAD_COLOR)

def request_inference(stub: system_pb2_grpc.DarkNetStub, frame: system_pb2.Image):
    inference = stub.make_inference(frame)
    print(inference)

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel("localhost:4000") as channel1:
        stub = system_pb2_grpc.CameraStub(channel1)
        frame = request_picture(stub)
    with grpc.insecure_channel("localhost:4002") as channel2:
        stub2 = system_pb2_grpc.DarkNetStub(channel2)
        print("-------------- GetFeature --------------")
        request_inference(stub2, frame)
    


if __name__ == "__main__":
    run()