import blinks
import glob
import path
import pickle
import random
from matplotlib import pyplot as plt


'''Plot each possible blink epoch to 50 X 50 plot for classification.
will probably not use this, but it was an option to use as basis for classification.
'''

# name = input('name: ')
name = 'martijn'
end = '_' + name + '.classification'

fig = plt.figure(figsize=(5,5),dpi=10)
fn = glob.glob(path.blinks + '*.blinks')
i = 0
for f in fn:
		print('plotting:',f)
		fin = open(f,'rb')
		b = pickle.load(fin)
		i = 1
		for p in b.peaks:
			print('#',i,b.fn.split('/')[-1])
			plt.clf()
			a = fig.add_axes([0,0,1,1])
			a.set_axis_off()
			plt.plot(b.veog[p-500:p+500],color='black',linewidth=9)
			fout = path.plot_blinks + b.fn.strip('.blinks') + '_' + str(i) + '.png'
			plt.savefig(fout,bbox_inches='tight',pad_inches=0)
