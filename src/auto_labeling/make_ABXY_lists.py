filename = '../../src/calib/ABback.txt'
A = []
B = []
xt = []
yt = []

with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        if line.endswith('.jpg'):
            filename = line
        elif line.endswith(')'):
            A_value, B_value = line.strip('()\n').split(',')
            line = file.readline()
            xt_value, yt_value, _ = map(float, line.split(';'))
            A.append(float(A_value))
            B.append(float(B_value))
            xt.append(xt_value)
            yt.append(yt_value)

print("A:", A)
print("B:", B)
print("xt:", xt)
print("yt:", yt)
