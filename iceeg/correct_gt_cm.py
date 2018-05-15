'''Correction script.
ground truth for channel model evaluation was incorrectly made
the info matrix of nsample X nchannels was flattened by np.ravel
the matrix becomes a concatenation of rows
I stepped through the data on a column basis (per channel) and therefore the
ground truth should be a concatenation of columns
this can be achieved with np.ravel(data, 'F')
this is corrected in model_ch_cnn file.

The old confusion matrices and reports should be corrected and I will do this
with this script

this script is single serving, and if you are unsure whether to use it than don't!
'''

import glob
import numpy as np
import path
import sklearn.metrics

def main():
	gt_fn = glob.glob(path.model_channel+'*gt.npy')
	p_fn = [f.replace('gt.npy','predicted.npy') for f in gt_fn]
	r_fn = [f.replace('gt.npy','report.txt') for f in gt_fn]
	cm_fn = [f.replace('gt.npy','cm.npy') for f in gt_fn]

	print('hallo')

	for i,f in enumerate(gt_fn):
		print(gt_fn[i])
		print(p_fn[i])
		print(r_fn[i])
		print(cm_fn[i])

		gtf = gt_fn[i]
		pf = p_fn[i]
		rf = r_fn[i]
		cmf = cm_fn[i]

		gt = np.load(gtf)
		gt = np.reshape(gt,[-1,26])
		gt = np.ravel(gt,'F')
		p = np.load(pf)
		cm = sklearn.metrics.confusion_matrix(gt,p)
		r = sklearn.metrics.classification_report(gt,p)

		print(r)
		print(" ")
		print(cm)
		print('\n\n\n')
		

		fout = open(rf,'w')
		fout.write(r)
		fout.close()

		np.save(gtf,gt)
		np.save(cmf,cm)

	 


