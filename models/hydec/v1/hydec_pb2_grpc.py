# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import hydec_pb2 as hydec__pb2


class HyDecStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getDecomposition = channel.unary_unary(
                '/hydec.HyDec/getDecomposition',
                request_serializer=hydec__pb2.DecompRequest.SerializeToString,
                response_deserializer=hydec__pb2.Decomposition.FromString,
                )
        self.getLayers = channel.unary_unary(
                '/hydec.HyDec/getLayers',
                request_serializer=hydec__pb2.DecompRequest.SerializeToString,
                response_deserializer=hydec__pb2.DecompositionLayers.FromString,
                )
        self.getDecompositionWithFile = channel.stream_unary(
                '/hydec.HyDec/getDecompositionWithFile',
                request_serializer=hydec__pb2.DecompFileRequest.SerializeToString,
                response_deserializer=hydec__pb2.Decomposition.FromString,
                )
        self.getLayersWithFile = channel.stream_unary(
                '/hydec.HyDec/getLayersWithFile',
                request_serializer=hydec__pb2.DecompFileRequest.SerializeToString,
                response_deserializer=hydec__pb2.DecompositionLayers.FromString,
                )


class HyDecServicer(object):
    """Missing associated documentation comment in .proto file."""

    def getDecomposition(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getLayers(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getDecompositionWithFile(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getLayersWithFile(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HyDecServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getDecomposition': grpc.unary_unary_rpc_method_handler(
                    servicer.getDecomposition,
                    request_deserializer=hydec__pb2.DecompRequest.FromString,
                    response_serializer=hydec__pb2.Decomposition.SerializeToString,
            ),
            'getLayers': grpc.unary_unary_rpc_method_handler(
                    servicer.getLayers,
                    request_deserializer=hydec__pb2.DecompRequest.FromString,
                    response_serializer=hydec__pb2.DecompositionLayers.SerializeToString,
            ),
            'getDecompositionWithFile': grpc.stream_unary_rpc_method_handler(
                    servicer.getDecompositionWithFile,
                    request_deserializer=hydec__pb2.DecompFileRequest.FromString,
                    response_serializer=hydec__pb2.Decomposition.SerializeToString,
            ),
            'getLayersWithFile': grpc.stream_unary_rpc_method_handler(
                    servicer.getLayersWithFile,
                    request_deserializer=hydec__pb2.DecompFileRequest.FromString,
                    response_serializer=hydec__pb2.DecompositionLayers.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'hydec.HyDec', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HyDec(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def getDecomposition(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hydec.HyDec/getDecomposition',
            hydec__pb2.DecompRequest.SerializeToString,
            hydec__pb2.Decomposition.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getLayers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/hydec.HyDec/getLayers',
            hydec__pb2.DecompRequest.SerializeToString,
            hydec__pb2.DecompositionLayers.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getDecompositionWithFile(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/hydec.HyDec/getDecompositionWithFile',
            hydec__pb2.DecompFileRequest.SerializeToString,
            hydec__pb2.Decomposition.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getLayersWithFile(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/hydec.HyDec/getLayersWithFile',
            hydec__pb2.DecompFileRequest.SerializeToString,
            hydec__pb2.DecompositionLayers.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
