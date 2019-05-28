#
# import pandas as pd
# import numpy as np
#
# what = 'detector'
#
# f = open('me_time_{}.txt'.format(what), 'w+')
#
#
# data = pd.read_csv('detector_time.txt', sep=":", header=None, names=['detector','time'])
#
# for name, df in data.groupby(what):
#     f.write(name + ': ' +np.mean(df['time']))
#
#
# f.close()

det = {}
with open('me_time_detector.txt', 'r') as f1:
    for line in f1:
        det[line.split(':')[0]] = line.split(':')[1][1:-2]
        print(line.split(':')[1][1:-2])
