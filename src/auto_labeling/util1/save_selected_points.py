import matplotlib.pyplot as plt
xx = []
yy = []
with open('../GuidingRobot/src/calib/scen1.txt', 'r') as file:
    for line in file:
        x, y = line.split(',')
        xx.append(float(x))
        yy.append(float(y))
plt.plot(xx,yy)
plt.show()

