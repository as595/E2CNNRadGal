import os,sys

n = 8
outdir = 'dn'+str(n)+'lenet'
if not os.path.exists(outdir): os.mkdir(outdir)

for i in range(0,5):

    os.system('python frdeepf_dnlenet.py '+str(n)+' \n')
    os.system('mv frdeepf_dnlenet.csv '+outdir+'/frdeepf_dnlenet_{}.csv'.format(i))
    

