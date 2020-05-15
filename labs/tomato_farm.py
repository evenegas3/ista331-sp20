import pandas as pd, numpy as np, random

features = {}
features['shape'] = ['circle','circle','circle','circle','oval','oval','irregular',]
features['size'] = ['small','medium','medium','medium','large']
features['color'] = ['red','red','red','red','yellow','green','orange']

file =  open('tomatoes.csv','w')

file.write('shape,size,color,deformed_status\n')

print('growing tomatoes...')
for i in range(1000):
    shape = features['shape'][random.randint(0,6)]
    size = features['size'][random.randint(0,4)]
    color = features['color'][random.randint(0,6)]
    deformed_status = 0
    if shape == 'irregular':
        deformed_status = 1
    if size == 'small' or size == 'large':
        if shape == 'oval' or color == 'green' or color == 'yellow' or color == 'orange':
            deformed_status = 1
    file.write(shape + ',' + size + ',' + color + ',' + str(deformed_status) + '\n')
    
file.close()
print('done!')