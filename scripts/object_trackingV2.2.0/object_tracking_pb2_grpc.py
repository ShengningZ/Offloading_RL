# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import object_tracking_pb2 as object__tracking__pb2


class ObjectDetectionServiceStub(object):
    """The object detection service definition.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessFrame = channel.unary_unary(
                '/objectdetection.ObjectDetectionService/ProcessFrame',
                request_serializer=object__tracking__pb2.FrameRequest.SerializeToString,
                response_deserializer=object__tracking__pb2.FrameResponse.FromString,
                )


class ObjectDetectionServiceServicer(object):
    """The object detection service definition.
    """

    def ProcessFrame(self, request, context):
        """Sends a frame to the edge server for processing
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessFrame': grpc.unary_unary_rpc_method_handler(
                    servicer.ProcessFrame,
                    request_deserializer=object__tracking__pb2.FrameRequest.FromString,
                    response_serializer=object__tracking__pb2.FrameResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'objectdetection.ObjectDetectionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ObjectDetectionService(object):
    """The object detection service definition.
    """

    @staticmethod
    def ProcessFrame(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/objectdetection.ObjectDetectionService/ProcessFrame',
            object__tracking__pb2.FrameRequest.SerializeToString,
            object__tracking__pb2.FrameResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)