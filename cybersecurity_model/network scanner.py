import socket
from ipaddress import ip_network, ip_address

def scan_network(network):
    print(f"Scanning network: {network}")
    for ip in ip_network(network, strict=False):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)  # Timeout for faster scanning
                s.connect((str(ip), 80))  # Check if port 80 (HTTP) is open
                print(f"[+] Host is up: {ip}")
        except (socket.timeout, ConnectionRefusedError):
            pass  # Skip if no response or connection refused
        except Exception as e:
            print(f"[!] Error scanning {ip}: {e}")

if __name__ == "__main__":
    network_input = input("Enter the network (e.g., 192.168.1.0/24): ")
    try:
        scan_network(network_input)
    except ValueError:
        print("Invalid network format. Use CIDR notation, e.g., 192.168.1.0/24.")
