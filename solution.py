import matplotlib.pyplot as plt
import pylab
import numpy
import math
a1x = [0, 1, 3]
a1y = [2, 0, 1]
a2x = [2, 3, 5]
a2y = [2, -2, 1]

def normal (k, mx, my, start, stop):
    x = []
    y = []

    for i in numpy.arange(start, stop, 0.01):
        x.append(i)
        if k == math.inf:
            y.append(my)
        else:
            y0 = - (i - mx) / k + my
            y.append(y0)


    return x, y
#граница между точками по правилу ближайшего соседа

def zipline (x1, y1, x2, y2):
    #м - середина отрезка
    mx = (x1+x2)/2
    my = (y1+y2)/2
    #нормаль
    if (x2 - x1) == 0:
        k = math.inf
    else:
        k = (y2 - y1) / (x2 - x1)

    return mx, my, k

def solvesystem (k1, k2, mx1, my1, mx2, my2):
    xy = numpy.array([[1., k1], [1., k2]])
    c = numpy.array([(mx1 + k1*my1), (mx2 + k2*my2)])
    numpy.linalg.solve(xy, c)
    return float(numpy.linalg.solve(xy, c)[0]), float(numpy.linalg.solve(xy, c)[1])



m14x, m14y, k14 = zipline(a1x[0], a1y[0], a2x[0], a2y[0])
m34x, m34y, k34 = zipline(a1x[2], a1y[2], a2x[0], a2y[0])
m24x, m24y, k24 = zipline(a1x[1], a1y[1], a2x[0], a2y[0])
m36x, m36y, k36 = zipline(a1x[2], a1y[2], a2x[2], a2y[2])
m35x, m35y, k35 = zipline(a1x[2], a1y[2], a2x[1], a2y[1])
m25x, m25y, k25 = zipline(a1x[1], a1y[1], a2x[1], a2y[1])
xs, ys = solvesystem(k14, k34, m14x, m14y, m34x, m34y)

start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)
print(start1424, stop1424)
start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)
start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)
start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)
start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)
start1424, stop1424 = solvesystem (k14, k24, m14x, m14y, m24x, m24y)






plt.figure(1)
plt.scatter(a1x, a1y, color='pink')
plt.scatter(a2x, a2y, color='green')
x24, y24 =  normal (k24, m24x, m24y, start1424, stop1424)
    #plt.plot(x, normal (k14, m14x, m14y) )
#plt.axvline((a2x[0]-a1x[0])/2 + a1x[0])
#plt.plot(x, normal (k34, m34x, m34y) )
plt.plot(x24, y24)
    #plt.plot(x, normal (k36, m36x, m36y) )
#plt.axvline( (a2x[2]-a1x[2])/2 + a1x[2])
#plt.plot(x, normal (k35, m35x, m35y) )
#plt.plot(x, normal (k25, m25x, m25y) )
plt.xlabel('x')
plt.ylabel('y')
pylab.legend(("a1", "a2"))

plt.show()


