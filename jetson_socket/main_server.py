from MacServer import MacServer
import time
import numpy as np
import pandas as pd

def main():
    time_received = []
    records_list = []  # To store all incoming records

    address = ['100.86.4.56', 4044]
    server = MacServer(address[0], int(address[1]))

    try:
        server.create_server()
        server.server_listen()
        server.connect()
    except Exception as e:
        print(f"Error setting up server: {e}")
        return

    while True:
        try:
            # Receive data from the client
            data = server.get_data()
            received_time = time.time()
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

        if data is not None:
            # Handle multiple records sent in a single batch separated by newlines
            records = data.strip().split("\n")
            for record in records:
                if record.strip():  # Skip empty lines
                    if record.strip() == "EXIT":
                        print("Received EXIT command. Closing server.")
                        server.close_sockets()
                        # Convert the collected records to a DataFrame and save as Excel
                        df = pd.DataFrame(records_list, 
                                          columns=["send_time", "time_process", "image_count", "time_received", "pred"])
                        df.to_excel("data_1.xlsx", index=False)
                        return

                    try:
                        # The record format: send_time,time_process,image_count,pred_str
                        # Split into 4 parts
                        parts = record.split(",", 3)
                        if len(parts) < 4:
                            raise ValueError(f"Record does not contain enough parts. {len(parts)} parts found. >>> \n{parts}")

                        send_time_str, time_process_str, image_count_str, pred = parts

                        send_time = float(send_time_str)
                        time_process = float(time_process_str)
                        image_count = float(image_count_str)
                        pred = np.frombuffer(pred, dtype=np.float32)

                        # Store record in the list
                        records_list.append([send_time, time_process, image_count, received_time, pred])

                    except ValueError as e:
                        print(f"Malformed record skipped: ({e})")

            time_received.append(time.time())

    # If we exit the loop (error or otherwise), ensure we close the server and save what we have.
    server.close_sockets()
    if records_list:
        df = pd.DataFrame(records_list, 
                          columns=["send_time", "time_process", "image_count", "time_received", "pred"])
        df.to_excel("data_1.xlsx", index=False)
    print("Server closed.")

if __name__ == "__main__":
    main()