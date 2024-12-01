from MacServer import MacServer
import time
from pandas import DataFrame

def main():    
    time_received = []
    time_sent = []
    process_time = []
    
    address = input("Enter Server Address: ").strip().split(':')
    
    server = MacServer(address[0], int(address[1]))
    
    try:
        server.create_server()
        server.server_listen()
        server.connect()
        
        while True:
            data = server.get_data()
            
            '''
            data[0] -> sent_time
            data[1] -> processing time
            data[2] -> prediction
            '''
            
            data = data.split("|")
            time_received.append(time.time())
            time_sent.append(data[0])
            process_time.append(data[1])
            
            preds = data[2]
            
            if data[1] == "exit_server":
                print("Exiting...")
                server.close_sockets()
                break
        
        df = DataFrame({"Time Send": time_sent, "Time Received": time_received})
        df.to_csv("data.csv", index = False)
        print("Data Written")
        exit()
    finally:
        server.close_sockets()
        exit()
        
if __name__ == "__main__":
    main()