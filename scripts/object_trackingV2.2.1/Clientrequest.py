import requests
import base64

# Example function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# URL of the server endpoint
url = 'http://localhost:5000/api/background-subtraction'

# Assuming you have an image file you want to process
image_path = 'path/to/your/image.jpg'
encoded_image = encode_image_to_base64(image_path)

# Make a POST request to the server
response = requests.post(url, json={"data": encoded_image})

print(response.json())