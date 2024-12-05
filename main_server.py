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
    with open("data.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["send_time", "time_process", "image_count", "time_received"])  # Updated to reflect data structure

        while True:
            try:
                data = server.get_data()
                recieved_time = time.time()
            except Exception as e:
                print(f"Error: {e}")
                break
            
            if data is not None:
                print(f"{data} received at {recieved_time}")
                ''''
                data->
                send_time, time_process, image_count, time_received
                '''
                # Split data if concatenated using a known delimiter (e.g., newline)
                records = data.split("\n")
                for record in records:
                    if record.strip():  # Skip empty lines
                        if record.strip() == "EXIT":
                            print("Malformed record skipped: EXIT")
                            break  # Exit the loop when EXIT is received

                        try:
                            # Assuming data format: send_time, time_process, image_count
                            parts = record.split(",")
                            send_time, time_process, image_count = map(float, parts[:3])
                            csv_writer.writerow([send_time, time_process, image_count, recieved_time])
                        except ValueError:
                            print(f"Malformed record skipped: {record}")
            
                time_received.append(time.time())
    
    server.close_sockets()

if __name__ == "__main__":
    main()