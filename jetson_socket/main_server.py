from MacServer import MacServer
import time
import numpy as np
import pandas as pd
import base64

def main():
    time_received = []
    records_list = []

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
            records = data.strip().split("\n")
            for record in records:
                if record.strip():
                    if record.strip() == "EXIT":
                        print("Received EXIT command. Closing server.")
                        server.close_sockets()
                        df = pd.DataFrame(records_list, 
                                          columns=["image_count", "pred", "time_process", "time_sent", "time_received", "transmission_time"])
                        df.to_excel("tranmission_processing_delay.xlsx", index=False)
                        return

                    try:
                        # The record format: send_time,time_process,image_count,pred_str
                        parts = record.split(",", 3)
                        if len(parts) < 4:
                            raise ValueError(f"Record does not contain enough parts. {len(parts)} parts found. >>> \n{parts}")

                        image_count, pred, time_process, send_time = parts

                        send_time, time_process = float(send_time), float(time_process)
                        image_count = int(image_count)
                    
                        pred = base64.b64decode(pred.encode('utf-8'))
                        pred = np.frombuffer(pred, dtype=np.float32)

                        records_list.append([image_count, pred, time_process, send_time, received_time, received_time - send_time])
                        
                        print(f"Image {image_count} received at {received_time}")
                        
                        if image_count == 399:
                            df = pd.DataFrame(records_list, 
                                              columns=["image_count", "pred", "time_process", "time_sent", "time_received", "transmission_time"])
                            print("All images received.")
                            df.to_excel("tranmission_processing_delay.xlsx", index=False)
                            print("Data saved to tranmission_processing_delay.xlsx")
                            server.close_sockets()
                            return     

                    except ValueError as e:
                        print(f"Malformed record skipped: ({e})")

            time_received.append(time.time())

if __name__ == "__main__":
    main()