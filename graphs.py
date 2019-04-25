import matplotlib.pyplot as plt

x3 = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
y3 = [512, 157, 105, 102, 60.15, 35.52, 24.7, 19.43, 17.86, 14.13]
plt.xlabel('minimal support')
plt.ylabel('time(s)')
plt.title('time consumption based on minimal support')
plt.plot(x3, y3)
plt.savefig('graph1.png')
plt.show()


x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y2 = [19.94, 20.17, 22.47, 23.54, 25.79, 25.84, 26.03, 31.37, 33.36, 34.32, 32.41, 31.88, 32.62, 33.78, 35.86, 36.95]
plt.xlabel('maximal length')
plt.ylabel('time(s)')
plt.plot(x2, y2)
#plt.show()
plt.savefig('graph2.png')

