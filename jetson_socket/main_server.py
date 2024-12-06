from MacServer import MacServer
import time
import csv

def main():
    time_received = []

    address = ['100.86.4.56', 4044]
    server = MacServer(address[0], int(address[1]))

    try:
        server.create_server()
        server.server_listen()
        server.connect()
    except Exception as e:
        print(f"Error setting up server: {e}")
        return

    # Open the CSV file for writing
    with open("data_1.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the CSV header
        csv_writer.writerow(["send_time", "time_process", "image_count", "time_received", "pred"])

        while True:
            try:
                # Receive data from the client
                data = server.get_data()
                received_time = time.time()
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

            if data is not None:
                # Handle multiple records sent as a single string with newline separators
                records = data.strip().split("\n")
                for record in records:
                    if record.strip():  # Skip empty lines
                        if record.strip() == "EXIT":
                            print("Received EXIT command. Closing server.")
                            server.close_sockets()
                            return

                        try:
                            # Split the record into parts
                            parts = record.split(",", 3)  # Split into 4 parts: send_time, time_process, image_count, pred
                            if len(parts) < 4:
                                raise ValueError("Record does not contain enough parts.")

                            # Extract the first three numeric parts
                            send_time, time_process, image_count = map(float, parts[:3])

                            # Extract the pred part as a string
                            pred = parts[3]  # Already string format

                            # Write the row to the CSV file
                            csv_writer.writerow([send_time, time_process, image_count, received_time, pred])

                        except ValueError as e:
                            print(f"Malformed record skipped: {record} ({e})")

                time_received.append(time.time())

    # Close the server sockets when exiting
    server.close_sockets()
    print("Server closed.")

if __name__ == "__main__":
    main()