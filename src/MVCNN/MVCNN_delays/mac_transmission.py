from MacServer import MacServer
import pandas as pd
import base64
import os
import time
import numpy as np

def main():
    address = ['100.86.4.56', 4044]
    server =  MacServer(address[0], int(address[1]))
    try:
        server.create_server()
        server.server_listen()
        server.connect()
    except Exception as e:
        print(f"Error setting up server: {e}")
        return
    
    records = []
    
    df = pd.DataFrame(
        records,
        columns=['send_time', 'receive_time', 'feature_map']
    )
    
    while True:
        data = server.get_data()
        if data == 'exit':
            print("Closing Server")
            server.close_sockets()
            df.to_excel("svcnn_mvcnn.xlsx", index=True)
            os._exit(0)
        try:
            data_rev = data.split("|")
            if len(data_rev) != 2:
                print(f"Received data in unexpected format: {len(data_rev)}")
                continue
            
            send_time = data_rev[0]
            feature_map = data_rev[1]
            receive_time = time.time()
            
            # Add padding if necessary
            missing_padding = len(feature_map) % 4
            if missing_padding:
                feature_map += '=' * (4 - missing_padding)
            try:
                feature_map = base64.b64decode(feature_map.encode('utf-8'))
                feature_map = np.frombuffer(feature_map, dtype=np.float32)
            except Exception as decode_error:
                print(f"Error decoding feature map: {decode_error}")
                continue
            df = df.(records,
                columns=['send_time', 'receive_time', 'feature_map']
                {
                    'send_time': send_time,
                    'receive_time': receive_time,
                    'feature_map': feature_map
                },
            )
        except Exception as e:
            print(f"Error processing data: {e}")
            break

if __name__ == "__main__":
    main()