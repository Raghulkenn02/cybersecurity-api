from scapy.all import *
import os

# Define the interface to capture packets on
INTERFACE = "wlan0"  # Change this to your network interface

# Define a function to analyze packets
def analyze_packet(packet):
    if packet.haslayer(IP):
        ip_src = packet[IP].src
        ip_dst = packet[IP].dst
        protocol = packet[IP].proto

        # Check for common IoT protocols
        if packet.haslayer(TCP):
            if packet[TCP].dport == 1883:
                print(f"[MQTT] Source: {ip_src} -> Destination: {ip_dst}")
            elif packet[TCP].dport == 5683:
                print(f"[CoAP] Source: {ip_src} -> Destination: {ip_dst}")
            elif packet[TCP].dport == 80 or packet[TCP].dport == 443:
                print(f"[HTTP/HTTPS] Source: {ip_src} -> Destination: {ip_dst}")

        # Print general information about the packet
        print(f"[INFO] Source: {ip_src}, Destination: {ip_dst}, Protocol: {protocol}")

# Start capturing packets
def start_capture():
    print(f"Starting packet capture on {INTERFACE}...")
    sniff(iface=INTERFACE, prn=analyze_packet, store=0)

if __name__ == "__main__":
    try:
        start_capture()
    except KeyboardInterrupt:
        print("\nPacket capture stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")