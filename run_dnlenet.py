import os,sys

for i in range(0,5):

    os.system('python frdeepf_dnlenet.py')
    os.system('mv frdeepf_dnlenet.csv dn4lenet/frdeepf_dnlenet_{}.csv'.format(i))
    

