from MacServer import MacServer
import time
import csv

def main():    
    time_received = []
    
    address = ['100.86.4.56', 4044]
    
    server = MacServer(address[0], int(address[1]))
    
    server.create_server()
    server.server_listen()
    server.connect()
    
    # Open the CSV file for writing
    with open("data_1.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["send_time", "time_process", "image_count", "time_received", "pred"])

        while True:
            try:
                data = server.get_data()
                recieved_time = time.time()
            except Exception as e:
                print(f"Error: {e}")
                break
            
            if data is not None:
                # print(f"{data} received at {recieved_time}")
                ''''
                data->
                send_time, time_process, image_count, time_received, pred
                '''
                records = data.split("\n")
                for record in records:
                    if record.strip():  # Skip empty lines
                        if record.strip() == "EXIT":
                            print("Malformed record skipped: EXIT")
                            server.close_sockets()
                            return

                        try:
                            parts = record.split(",")
                            print(len(parts))
                            send_time, time_process, image_count = map(float, parts[:3])
                            pred = str(parts[3])
                            csv_writer.writerow([send_time, time_process, image_count, recieved_time, pred])
                        except ValueError:
                            print(f"Malformed record skipped: {record}")
            
                time_received.append(time.time())
    
    server.close_sockets()

if __name__ == "__main__":
    main()