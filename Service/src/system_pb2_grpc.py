# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import system_pb2 as system__pb2


class NodeManagerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.addRecipeNode = channel.unary_unary(
                '/NodeManager/addRecipeNode',
                request_serializer=system__pb2.Node.SerializeToString,
                response_deserializer=system__pb2.Node.FromString,
                )
        self.removeNode = channel.unary_unary(
                '/NodeManager/removeNode',
                request_serializer=system__pb2.NodeId.SerializeToString,
                response_deserializer=system__pb2.Node.FromString,
                )
        self.loadRecipe = channel.unary_unary(
                '/NodeManager/loadRecipe',
                request_serializer=system__pb2.Recipe.SerializeToString,
                response_deserializer=system__pb2.NoneArgs.FromString,
                )


class NodeManagerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def addRecipeNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def removeNode(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def loadRecipe(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NodeManagerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'addRecipeNode': grpc.unary_unary_rpc_method_handler(
                    servicer.addRecipeNode,
                    request_deserializer=system__pb2.Node.FromString,
                    response_serializer=system__pb2.Node.SerializeToString,
            ),
            'removeNode': grpc.unary_unary_rpc_method_handler(
                    servicer.removeNode,
                    request_deserializer=system__pb2.NodeId.FromString,
                    response_serializer=system__pb2.Node.SerializeToString,
            ),
            'loadRecipe': grpc.unary_unary_rpc_method_handler(
                    servicer.loadRecipe,
                    request_deserializer=system__pb2.Recipe.FromString,
                    response_serializer=system__pb2.NoneArgs.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'NodeManager', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NodeManager(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def addRecipeNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NodeManager/addRecipeNode',
            system__pb2.Node.SerializeToString,
            system__pb2.Node.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def removeNode(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NodeManager/removeNode',
            system__pb2.NodeId.SerializeToString,
            system__pb2.Node.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def loadRecipe(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NodeManager/loadRecipe',
            system__pb2.Recipe.SerializeToString,
            system__pb2.NoneArgs.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class CameraStub(object):
    """! ====================================================================
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.take_picture = channel.unary_unary(
                '/Camera/take_picture',
                request_serializer=system__pb2.NoneArgs.SerializeToString,
                response_deserializer=system__pb2.Image.FromString,
                )


class CameraServicer(object):
    """! ====================================================================
    """

    def take_picture(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CameraServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'take_picture': grpc.unary_unary_rpc_method_handler(
                    servicer.take_picture,
                    request_deserializer=system__pb2.NoneArgs.FromString,
                    response_serializer=system__pb2.Image.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Camera', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Camera(object):
    """! ====================================================================
    """

    @staticmethod
    def take_picture(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Camera/take_picture',
            system__pb2.NoneArgs.SerializeToString,
            system__pb2.Image.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ObjectDetectionStub(object):
    """! ====================================================================
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.make_inference = channel.unary_unary(
                '/ObjectDetection/make_inference',
                request_serializer=system__pb2.Image.SerializeToString,
                response_deserializer=system__pb2.InferenceResult.FromString,
                )
        self.loadNetWork = channel.unary_unary(
                '/ObjectDetection/loadNetWork',
                request_serializer=system__pb2.NetworkConfig.SerializeToString,
                response_deserializer=system__pb2.RequestStatus.FromString,
                )


class ObjectDetectionServicer(object):
    """! ====================================================================
    """

    def make_inference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def loadNetWork(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'make_inference': grpc.unary_unary_rpc_method_handler(
                    servicer.make_inference,
                    request_deserializer=system__pb2.Image.FromString,
                    response_serializer=system__pb2.InferenceResult.SerializeToString,
            ),
            'loadNetWork': grpc.unary_unary_rpc_method_handler(
                    servicer.loadNetWork,
                    request_deserializer=system__pb2.NetworkConfig.FromString,
                    response_serializer=system__pb2.RequestStatus.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ObjectDetection', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ObjectDetection(object):
    """! ====================================================================
    """

    @staticmethod
    def make_inference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ObjectDetection/make_inference',
            system__pb2.Image.SerializeToString,
            system__pb2.InferenceResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def loadNetWork(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ObjectDetection/loadNetWork',
            system__pb2.NetworkConfig.SerializeToString,
            system__pb2.RequestStatus.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ObjectPositioningStub(object):
    """! ====================================================================
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.make_worldmapping = channel.unary_unary(
                '/ObjectPositioning/make_worldmapping',
                request_serializer=system__pb2.WorldMappingRequest.SerializeToString,
                response_deserializer=system__pb2.WorldMappingResult.FromString,
                )


class ObjectPositioningServicer(object):
    """! ====================================================================
    """

    def make_worldmapping(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectPositioningServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'make_worldmapping': grpc.unary_unary_rpc_method_handler(
                    servicer.make_worldmapping,
                    request_deserializer=system__pb2.WorldMappingRequest.FromString,
                    response_serializer=system__pb2.WorldMappingResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ObjectPositioning', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ObjectPositioning(object):
    """! ====================================================================
    """

    @staticmethod
    def make_worldmapping(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ObjectPositioning/make_worldmapping',
            system__pb2.WorldMappingRequest.SerializeToString,
            system__pb2.WorldMappingResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
