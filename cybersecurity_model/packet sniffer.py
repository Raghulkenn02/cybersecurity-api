import socket
import os
from datetime import datetime

# 1. Log Packets to a File
def log_packet(packet, log_dir="logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"sniffer_log_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(log_file, "a") as f:
        f.write(f"{datetime.now()} - {packet}\n")

# 2. Sniff Packets
def start_sniffer(interface=None):
    print("[INFO] Starting packet sniffer...")
    try:
        # Create a raw socket
        sniffer = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
        if interface:
            sniffer.bind((interface, 0))  # Bind to specific interface

        print(f"[INFO] Listening on {interface if interface else 'all interfaces'}...")
        while True:
            raw_packet, addr = sniffer.recvfrom(65535)
            packet_data = raw_packet.hex()
            print(f"[INFO] Packet received from {addr}: {packet_data[:100]}...")  # Display first 100 chars
            log_packet(packet_data)

    except KeyboardInterrupt:
        print("[INFO] Stopping sniffer...")
    except PermissionError:
        print("[ERROR] Permission denied. Try running with elevated privileges (e.g., sudo).")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

# Main Function
if __name__ == "__main__":
    print("=== Simple Packet Sniffer ===")
    interface = input("Enter the network interface to sniff on (leave blank for all interfaces): ").strip()
    start_sniffer(interface if interface else None)


