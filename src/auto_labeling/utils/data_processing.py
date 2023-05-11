import re

# Create empty lists
X = []
Y = []
A = []
B = []

# Read file line by line
with open('../../calib/B2.txt', 'r') as f:
    lines = f.readlines()  # Read all lines from file

    for line in lines:
        values = line.split()
        # Extract values from strings and add to lists
        if 'X' in values:
            X.append(float(values[7]))
            Y.append(float(values[9]))
        if 'A' in values:
            A.append(float(values[7]))
            B.append(float(values[9]))

        # Print the resulting lists
    print(X)
    print(Y)
    print(A)
    print(B)




