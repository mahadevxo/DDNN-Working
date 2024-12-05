import socket

class MacServer:
    def __init__(self, mac_ip, mac_port):
        self.mac_ip = mac_ip
        self.mac_port = mac_port
        self.mac_socket = None
        self.jetson_socket = None
        self.buffer = b""

    def create_server(self):
        self.mac_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mac_socket.bind((self.mac_ip, self.mac_port))
        print(f"Server socket created and bound to {self.mac_ip}:{self.mac_port}")

    def server_listen(self):
        if self.mac_socket:
            self.mac_socket.listen(5)
            print("Server is listening for incoming connections...")
        else:
            print("Server socket not created. Call create_server first.")

    def connect(self):
        if self.mac_socket:
            self.jetson_socket, jetson_address = self.mac_socket.accept()
            print(f"Connected to client at {jetson_address}")
        else:
            print("Server socket not created. Call create_server first.")

    def get_data(self):
        if self.jetson_socket:
            while True:
                try:
                    chunk = self.jetson_socket.recv(1024)
                    if not chunk:
                        return None
                    self.buffer += chunk
                    
                    if b"," in self.buffer:
                        message, self.buffer = self.buffer.split(b"\n", 1)
                        return message.decode('utf-8')
                except Exception as e:
                    print(f"Error receiving data: {e}")
                    return None
        else:
            print("No client connection established.")
            return None

    def close_sockets(self):
        if self.jetson_socket:
            self.jetson_socket.close()
            print("Client socket closed.")
        if self.mac_socket:
            self.mac_socket.close()
            print("Server socket closed.")