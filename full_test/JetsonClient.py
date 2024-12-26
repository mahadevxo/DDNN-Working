import socket

class JetsonClient:
    def __init__(self, mac_ip, mac_port):
        self.mac_ip = mac_ip
        self.mac_port = mac_port
        self.jetson_socket = None

    def connect_to_server(self):
        self.jetson_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.jetson_socket.connect((self.mac_ip, self.mac_port))
        print(f"Connected to server at {self.mac_ip}:{self.mac_port}")

    def send_data(self, message):
        if self.jetson_socket:
            self.jetson_socket.send((message + "\n").encode())
            # print("Data sent")
        else:
            print("Not connected to the server.")

    def close_connection(self):
        if self.jetson_socket:
            self.jetson_socket.close()
            print("Connection closed.")
            exit()
        else:
            print("No active connection to close.")