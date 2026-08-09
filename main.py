from sender import send_command


print("Fusion Gesture Controller")
print("A : Left Rotate")
print("D : Right Rotate")
print("Q : Quit")


while True:

    key = input("> ")


    if key == "a":
        send_command("rotate_left")


    elif key == "d":
        send_command("rotate_right")


    elif key == "q":
        break