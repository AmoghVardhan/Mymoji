# import os
#
# fifo_name = 'fifo'
#
# def main():
#     data = 'hello world'
#     fifo = open(fifo_name, 'w')
#     fifo.write(data)
#     fifo.close()
# if __name__ == '__main__':
#     main()


import socket
import sys

HOST, PORT = "192.168.0.105", 27015

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect((HOST, PORT))
s.send(b'9')
