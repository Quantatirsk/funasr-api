"""
Client Web Server for funasr-api

Originally inspired by:
  FunASR-Nano-2512 Client Web Server
  Author: 凌封
  Source: https://aibook.ren (AI全书)

The current implementation has been significantly modified
and extended for this project.
Main modifications:
  - Adapted the client-side WebSocket logic to be compatible with
    Alibaba Cloud ASR WebSocket API
"""
import http.server
import socketserver
import os
import socket
import argparse
import ssl

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def main():
    parser = argparse.ArgumentParser(description="FunASR-Nano-2512 Client Web Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to serve on (default: 8001)")
    parser.add_argument("--cert", type=str, help="Path to SSL certificate file (for HTTPS)")
    parser.add_argument("--key", type=str, help="Path to SSL private key file (for HTTPS)")
    
    args = parser.parse_args()
    
    port = args.port
    cert_file = args.cert
    key_file = args.key
    
    use_https = cert_file and key_file
    
    ip = get_ip_address()
    protocol = "HTTPS" if use_https else "HTTP"
    
    print(f"Serving {protocol} on 0.0.0.0 port {port} ...")
    print(f"Server directory: {DIRECTORY}")
    if use_https:
        print(f"SSL Certificate: {cert_file}")
        print(f"SSL Private Key: {key_file}")
    
    print("\n" + "="*60)
    print(f"访问地址:")
    if use_https:
        print(f"  Local:   https://localhost:{port}")
        print(f"  Network: https://{ip}:{port}")
    else:
        print(f"  Local:   http://localhost:{port}")
        print(f"  Network: http://{ip}:{port}")
    print("="*60)
    print("="*60)
    print("\n[使用说明]")
    print("1. 【推荐】本地电脑测试：")
    if use_https:
        print(f"   请直接访问: https://localhost:{port}")
        print("   使用HTTPS，浏览器会允许麦克风权限。")
    else:
        print(f"   请直接访问: http://localhost:{port}")
        print("   无需任何配置，浏览器会直接允许麦克风权限。")
    
    print("\n2. 【高级】远程访问测试：")
    if use_https:
        print(f"   如果您通过 IP (https://{ip}:{port}) 访问，浏览器会允许麦克风权限。")
    else:
        print(f"   如果您通过 IP (http://{ip}:{port}) 访问，浏览器可能因非 HTTPS 禁止麦克风。")
        print("   解决办法: 在 Chrome 地址栏输入 chrome://flags/#unsafely-treat-insecure-origin-as-secure")
        print(f"   填入 http://{ip}:{port} 并启用。")
    print("="*60 + "\n")

    with socketserver.TCPServer(("", port), Handler) as httpd:
        if use_https:
            # 包装socket为SSL
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            print(f"SSL/TLS enabled with certificate: {cert_file}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    main()
