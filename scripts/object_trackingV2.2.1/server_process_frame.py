from concurrent import futures
import grpc
import object_tracking_pb2
import object_tracking_pb2_grpc
from object_detection import perform_detection  # Assuming this function exists

class ObjectDetectionServicer(object_tracking_pb2_grpc.ObjectDetectionServicer):
    def DetectObjects(self, request, context):
        image_data = request.image
        # Convert image data from bytes to an appropriate format if needed
        # Perform detection
        detected_objects = perform_detection(image_data)
        # Convert detection results to DetectionResponse
        response = object_tracking_pb2.DetectionResponse()
        for obj in detected_objects:
            detected_object = response.objects.add()
            detected_object.id = obj['id']
            detected_object.label = obj['label']
            detected_object.confidence = obj['confidence']
            detected_object.bbox.extend(obj['bbox'])
        return response

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    object_tracking_pb2_grpc.add_ObjectDetectionServicer_to_server(ObjectDetectionServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()