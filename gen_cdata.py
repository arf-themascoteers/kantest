import csv
from random import uniform

import numpy as np

with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x1', 'x2', 'y'])
    for _ in range(1000):
        x1 = uniform(-1, 1)
        x2 = uniform(-1, 1)
        y = np.zeros_like(x1)
        y = y[(x1>0) & (x2<0)]

        writer.writerow([x1, x2, y])