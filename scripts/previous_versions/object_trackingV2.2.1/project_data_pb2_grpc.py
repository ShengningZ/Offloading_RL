# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import project_data_pb2 as project__data__pb2


class VideoProcessingServiceStub(object):
    """gRPC 服务定义

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ProcessVideo = channel.stream_stream(
                '/project_data.VideoProcessingService/ProcessVideo',
                request_serializer=project__data__pb2.Image.SerializeToString,
                response_deserializer=project__data__pb2.DetectionResult.FromString,
                )


class VideoProcessingServiceServicer(object):
    """gRPC 服务定义

    """

    def ProcessVideo(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_VideoProcessingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ProcessVideo': grpc.stream_stream_rpc_method_handler(
                    servicer.ProcessVideo,
                    request_deserializer=project__data__pb2.Image.FromString,
                    response_serializer=project__data__pb2.DetectionResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'project_data.VideoProcessingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class VideoProcessingService(object):
    """gRPC 服务定义

    """

    @staticmethod
    def ProcessVideo(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/project_data.VideoProcessingService/ProcessVideo',
            project__data__pb2.Image.SerializeToString,
            project__data__pb2.DetectionResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ObjectDetectionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectObject = channel.unary_unary(
                '/project_data.ObjectDetectionService/DetectObject',
                request_serializer=project__data__pb2.Image.SerializeToString,
                response_deserializer=project__data__pb2.DetectionResult.FromString,
                )


class ObjectDetectionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectObject(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ObjectDetectionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectObject': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectObject,
                    request_deserializer=project__data__pb2.Image.FromString,
                    response_serializer=project__data__pb2.DetectionResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'project_data.ObjectDetectionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ObjectDetectionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectObject(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/project_data.ObjectDetectionService/DetectObject',
            project__data__pb2.Image.SerializeToString,
            project__data__pb2.DetectionResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class BackgroundSubtractionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ApplyBackgroundSubtraction = channel.unary_unary(
                '/project_data.BackgroundSubtractionService/ApplyBackgroundSubtraction',
                request_serializer=project__data__pb2.Image.SerializeToString,
                response_deserializer=project__data__pb2.ForegroundMask.FromString,
                )


class BackgroundSubtractionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ApplyBackgroundSubtraction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_BackgroundSubtractionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ApplyBackgroundSubtraction': grpc.unary_unary_rpc_method_handler(
                    servicer.ApplyBackgroundSubtraction,
                    request_deserializer=project__data__pb2.Image.FromString,
                    response_serializer=project__data__pb2.ForegroundMask.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'project_data.BackgroundSubtractionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class BackgroundSubtractionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ApplyBackgroundSubtraction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/project_data.BackgroundSubtractionService/ApplyBackgroundSubtraction',
            project__data__pb2.Image.SerializeToString,
            project__data__pb2.ForegroundMask.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class KalmanFilterServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.UpdateState = channel.unary_unary(
                '/project_data.KalmanFilterService/UpdateState',
                request_serializer=project__data__pb2.DetectionResult.SerializeToString,
                response_deserializer=project__data__pb2.StateUpdate.FromString,
                )


class KalmanFilterServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def UpdateState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_KalmanFilterServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'UpdateState': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateState,
                    request_deserializer=project__data__pb2.DetectionResult.FromString,
                    response_serializer=project__data__pb2.StateUpdate.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'project_data.KalmanFilterService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class KalmanFilterService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def UpdateState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/project_data.KalmanFilterService/UpdateState',
            project__data__pb2.DetectionResult.SerializeToString,
            project__data__pb2.StateUpdate.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)