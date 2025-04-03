import usb.core
import usb.util

# Replace these with your Morpho device's actual vendor and product IDs
VENDOR_ID = 0x079B  # Your vendor ID
PRODUCT_ID = 0x0047  # Your product ID

# Find the Morpho device
device = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)

if device is None:
    print("Morpho device not found.")
else:
    print("Morpho device found.")
    # Set the active configuration
    device.set_configuration()

    # You can also try to read from the device or send a command if applicable
    try:
        # Replace with the actual endpoint address and size
        endpoint = device[0][(0,0)][0]  # Get the first endpoint
        data = device.read(endpoint.bEndpointAddress, endpoint.wMaxPacketSize)
        print(f"Data read from device: {data}")
    except usb.core.USBError as e:
        print(f"Error reading from device: {e}")
        