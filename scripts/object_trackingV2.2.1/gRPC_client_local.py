import grpc
import object_tracking_pb2
import object_tracking_pb2_grpc


def run():
    channel = grpc.insecure_channel('server_address:50051')  # Replace with your server's address
    stub = object_tracking_pb2_grpc.ObjectDetectionStub(channel)
    # Convert your image to bytes
    image_data = ...  # This should be the image data in bytes
    response = stub.DetectObjects(object_tracking_pb2.Image(image=image_data))
    print("Detected Objects:")
    for obj in response.objects:
        print(f"Label: {obj.label}, Confidence: {obj.confidence}, BBox: {obj.bbox}")

if __name__ == '__main__':
    run()