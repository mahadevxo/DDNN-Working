from MacServer import MacServer
import time
from pandas import DataFrame


def main():    
    time_received = []
    
    address =['100.86.4.56', 4044]
    
    server = MacServer(address[0], int(address[1]))
    
    server.create_server()
    server.server_listen()
    server.connect()
    
    file_write = open("data.csv", "a")
    
    while True:
        try:
            data = server.get_data()
        except Exception as e:
            print(f"Error: {e}")
            break
        
        if data is not None:
            
            print(f"Received {time.time()}")
            
            file_write.write(f"{data}\n")
            
            time_received.append(time.time())
            
            if data[1] == "exit_server":
                print("Exiting...")
                server.close_sockets()
                break
    
    file_write.close()

if __name__ == "__main__":
    main()