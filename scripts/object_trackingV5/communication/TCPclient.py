import socket

def send_message(host='10.192.31.3', port=65432, message="Hello, Server"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(message.encode())
        data = s.recv(1024)
        print(f"Received: {data.decode()}")

if __name__ == "__main__":
    send_message(message="Hello from Client!")
