import hashlib
import os
import pefile

def calculate_hash(file_path):
    """Calculate MD5, SHA1, and SHA256 hashes for the file."""
    hashes = {'MD5': hashlib.md5(), 'SHA1': hashlib.sha1(), 'SHA256': hashlib.sha256()}
    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)  # Read file in chunks
                if not chunk:  # Exit loop if no more data
                    break
                for h in hashes.values():
                    h.update(chunk)
        return {name: h.hexdigest() for name, h in hashes.items()}
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def analyze_file(file_path):
    """Analyze the file and extract metadata."""
    try:
        pe = pefile.PE(file_path)
        print(f"File: {file_path}")
        print("---- PE Metadata ----")
        print(f"Entry Point: {hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)}")
        print(f"Image Base: {hex(pe.OPTIONAL_HEADER.ImageBase)}")
        print(f"Sections: {[section.Name.decode('utf-8').strip() for section in pe.sections]}")
        pe.close()
    except pefile.PEFormatError:
        print("Not a valid PE file or not a PE file at all.")

if __name__ == "__main__":
    file_path = input("Enter the file path for analysis: ")
    if os.path.exists(file_path):
        hashes = calculate_hash(file_path)
        if hashes:
            print("---- File Hashes ----")
            for name, value in hashes.items():
                print(f"{name}: {value}")
        analyze_file(file_path)
    else:
        print("File not found!")

