from MacServer import MacServer
import time

def main():    
    time_received = []
    
    address =['100.86.4.56', 4044]
    
    server = MacServer(address[0], int(address[1]))
    
    server.create_server()
    server.server_listen()
    server.connect()
    
    file_write = open("data.csv", "w")
    file_write.write("send_time, time_process, image_count, time_received\n")
    
    while True:
        try:
            data = server.get_data()
            
            if data is None:
                break
            
            recieved_time = time.time()
        except Exception as e:
            print(f"Error: {e}")
            break
        else:
            
            print(f"Received {recieved_time}")
            ''''
            data->
            send_time, time_process, image_count, time_received
            '''
            file_write.write(f"{data},{recieved_time}\n")
            
            time_received.append(time.time())
    
    file_write.close()
    server.close_connection()

if __name__ == "__main__":
    main()