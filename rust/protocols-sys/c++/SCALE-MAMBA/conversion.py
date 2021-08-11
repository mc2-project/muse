import numpy as np
import re
import os

# TODO: Make this easy script to run
# Metadata
layer_size = 300000
#prime = 2061584302081
#prime = 7340033
prime = 268369921
field_bits = len(bin(prime).lstrip('-0b'))
print("BITS: ", field_bits)
bit_repr = '{0:0%db}' % (field_bits)

# Helpers
def sample():
    return np.array([np.random.randint(prime)])

def sample_vec():
    return np.random.randint(prime, size=layer_size)

def share(arr):
    mask = sample_vec()
    share = (arr - mask) % prime
    return mask, share

def to_bits(l, l_bits):
    for num in l:
        num_bits = [int(c) for c in bit_repr.format(num)]
        for bit in reversed(num_bits):
            l_bits.append(bit)

# Sample inputs
a = sample()
b = sample()

Mr = sample_vec()
r = sample_vec()

labels = np.random.randint(prime, size=(layer_size*field_bits*4)) \
    .reshape(2, layer_size*field_bits*2)

# Compute MACs
br = np.array([(long(b[0]) * long(r_l)) % prime for r_l in r])
c_y, s_y = share(Mr)
ay = np.array([(long(a[0]) * long(c_y_l)) % prime for c_y_l in c_y])

# Share MACs
c_ay, s_ay = share(ay)
c_br, s_br = share(br)

# Decompose client input to bits
c_y_bits = list()
to_bits(c_y, c_y_bits)

r_bits = list()
to_bits(r, r_bits)

# Output all inputs to files for reading
client_inp_list = np.hstack((c_y_bits, r_bits, c_ay, c_br))
client_inp = " ".join(str(inp) for inp in client_inp_list)

server_inp_list = np.hstack((a, b, labels[0], labels[1]))
server_inp = " ".join(str(inp) for inp in server_inp_list)

with open("./Channels/chan_1.in", "w") as f:
    f.write(client_inp)

with open("./Channels/chan_0.in", "w") as f:
    f.write(server_inp)

# Remove old .out files
if os.path.exists("./Channels/chan_0.out"):
    os.remove("./Channels/chan_0.out")

if os.path.exists("./Channels/chan_1.out"):
    os.remove("./Channels/chan_1.out")

t = input("Input something once program finishes")

# Server extracts shares of output + garbled labels, and sends label shares to
# client if output correct
server_g = np.array([], dtype=np.int64)
with open("./Channels/chan_0.out", "r") as f:
    for i, line in enumerate(f):
        # On first line we check whether the values we received from client
        # are valid shares of 0
        values = filter(lambda x: re.match("[-]?\d+$", x), line.split(" "))
        values = [int(elem) % prime for elem in values]
        if i == 0:
            check = True
            for share in range(layer_size):
                check &= values[share] == s_ay[share]
                # TODO
                if not check:
                    print("FAIL")
                    print(values[share], s_ay[share])
            for share in range(layer_size):
                check &= values[layer_size + share] == s_br[share]
                # TODO
                if not check:
                    print("FAIL")
                    print(values[layer_size+share], s_br[share])
            if not check:
                raise Exception("CHECK FAILED :(")
            if values[layer_size*2] == 0:
                server_g = np.append(server_g, values[layer_size*2+1])
        if values[0] == 0:
            server_g = np.append(server_g, values[1])

# Client extracts garbled labels from output
client_g = np.array([], dtype=np.int64)
with open("./Channels/chan_1.out", "r") as f:
    for i, line in enumerate(f):
        values = filter(lambda x: re.match("[-]?\d+$", x), line.split(" "))
        values = [int(elem) % prime for elem in values]
        if values[0] == 1:
            client_g = np.append(client_g, values[1])

#for i in range(len(server_g)):
#    print(server_g[i], client_g[i], (server_g[i] + client_g[i]) % prime)

# Client reconstructs garbled labels from server share
result = (server_g + client_g) % prime
client_bits = np.hstack((c_y_bits, r_bits))


correct = True
for i, lab in enumerate(result):
    correct &= (labels[client_bits[i]][i] == result[i])
    if not correct:
        print(i, "EXPECTED: ", labels[client_bits[i]][i], "GOT: ", result[i])

print("Client labels correct: ", correct)
