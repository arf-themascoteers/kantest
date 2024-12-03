import csv
from random import uniform
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x1', 'x2', 'y'])
    for _ in range(100):
        x1 = uniform(-1, 1)
        x2 = uniform(-1, 1)
        y = x1 * x1 - 0.1 * x2
        writer.writerow([x1, x2, y])
