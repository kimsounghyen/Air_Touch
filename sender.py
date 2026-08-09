import socket
import json


HOST = "127.0.0.1"
PORT = 5000


def send_command(command):

    data = {
        "command": command
    }

    message = json.dumps(data)


    try:
        client = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        client.connect(
            (HOST, PORT)
        )

        client.send(
            message.encode()
        )

        client.close()


    except Exception as e:
        print(e)