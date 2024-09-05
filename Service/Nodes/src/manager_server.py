from core import system_pb2_grpc
from core import system_pb2
from core.core import serve, logging, LOG_LEVEL
from codetiming import Timer
import grpc
from interface.Nodes import grpcNode as Node



def request_picture(node):
    picture = node.stub.take_picture(system_pb2.NoneArgs())
    return {0:picture}

def request_inference(node):
    inference = node.stub.make_inference(node.inputs[0].value) #? node.inputs['frame']
    return {0:inference}

def request_worldmapping(node):   
    mapping = node.stub.make_worldmapping(
        system_pb2.WorldMappingRequest(
            image=node.inputs[0].value,
            inference=node.inputs[1].value
        )
    )
    
    return {0: mapping}

logger = logging.getLogger("NodeManagerService")
class NodeManagerServicer(system_pb2_grpc.NodeManagerServicer):
    def __init__(self):
        super().__init__()
        self.CallbackRegistry = {
            'take_picture': request_picture,
            'make_inference':request_inference,
            'make_worldmapping':request_worldmapping,
        }
        self.channels = {}
        self.NodeRegistry = {}
    
    @Timer("loadRecipe",  text="{name} elapsed time: {:.4f}s", logger=logger.debug)
    def loadRecipe(self, request, context):
        NodeRegistry = {}
        for step in range(2):   #create channels and nodes, update connections
            for node in request.nodes:
                match step:
                    case 0:
                        if node.channel not in self.channels:
                            self.channels[node.channel] = grpc.insecure_channel(f"localhost:{node.channel}")
                        NodeRegistry[node.id] = Node(
                                id = node.id,
                                connections=node.connections,
                                callback = self.CallbackRegistry[node.callback],
                                stub = getattr(system_pb2_grpc, f"{node.service}Stub")(self.channels[node.channel]),
                            )
                    case 1:
                        NodeRegistry[node.id].update_connections(NodeRegistry)
        self.NodeRegistry = NodeRegistry
        for _, node in zip(range(10), self.NodeRegistry['camera_node_000']):
            print("="*5,_,'|\t',node.name,'\t|','='*5)
            
        return system_pb2.NoneArgs()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="50051")
    
    args = parser.parse_args()

    servicer = NodeManagerServicer()
    serve(system_pb2_grpc.add_NodeManagerServicer_to_server, servicer, logger, server_port=args.service_port)