import socket

HOST = 'robotarm.local'  # The server's hostname or IP address
PORT = 1234        # The port used by the server

STEPS_PER_REVOLUTION = 200 * 16  # Change this to the number of steps per revolution for your motors

def angle_to_steps(angle):
    return int(STEPS_PER_REVOLUTION * (angle / 360))

def send_angles(angles, total_time):
    steps = [angle_to_steps(angle) for angle in angles]
    steps.append(total_time)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(' '.join(map(str, steps)).encode())
