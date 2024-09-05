import grpc
from core import system_pb2_grpc
from core import system_pb2

def loadRecipe(stub: system_pb2_grpc.NodeManagerStub):
    recipe = system_pb2.Recipe(
        id='1',
        nodes=[
            {
                'id': 'camera_node_000',
                'service': 'Camera',
                'channel':'4000',
                'callback': 'take_picture',
                'connections':[
                    {
                        'output':{
                            'parent':'camera_node_000',
                            'type':'OUTPUT',
                            'id':0
                        },
                        'input':{
                            'parent':'interpret_node_001',
                            'type':'INPUT',
                            'id':0
                        }
                    },
                    {
                        'output':{
                            'parent':'camera_node_000',
                            'type':'OUTPUT',
                            'id':0
                        },
                        'input':{
                            'parent':'inference_node_001',
                            'type':'INPUT',
                            'id':0
                        }
                    },
                ]
            },
            {
                'id': 'inference_node_001',
                'service': 'ObjectDetection',
                'channel':'4002',
                'callback': 'make_inference',
                'connections':[
                    {
                        'output':{
                            'parent':'inference_node_001',
                            'type':'OUTPUT',
                            'id':0
                        },
                        'input':{
                            'parent':'interpret_node_001',
                            'type':'INPUT',
                            'id':1
                        }
                    }
                ]
            },
            {
                'id': 'interpret_node_001',
                'service': 'ObjectPositioning',
                'channel':'4003',
                'callback': 'make_worldmapping',
                'connections':[
                    {
                        'output':{
                            'parent':'interpret_node_001',
                            'type':'OUTPUT',
                            'id':0
                        },
                        'input':{
                            'parent':'camera_node_000',
                            'type':'INPUT',
                            'id':0
                        }
                    }
                ]
            },
        ]
    )
    stub.loadRecipe(recipe)

def run(port):
    with grpc.insecure_channel(f"localhost:{port}") as channel1:
        stub = system_pb2_grpc.NodeManagerStub(channel1)
        loadRecipe(stub)

if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the server service are.", default="50051")
    
    args = parser.parse_args()
    
    run(args.service_port)