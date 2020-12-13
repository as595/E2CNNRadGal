import os,sys

for i in range(0,5):

    os.system('python frdeepf_lenet.py')
    os.system('mv frdeepf_lenet.csv lenet/frdeepf_lenet_{}.csv'.format(i))
    
