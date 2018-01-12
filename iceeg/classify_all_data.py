import glob
import model_deep_artifact as mda
import numpy as np
import os
import path



def main(model_object):
	fn = glob.glob(path.artifact_data_all_pp +'*data.npy')

	for f in fn:
		print(f)
		if os.path.isfile(f.replace('data','pred')):
			print(f,'already done!')
			continue
		pred = mda.data2prediction(f,model_object)
		print('Saving to:',f.replace('data','pred'))
		np.save(f.replace('data','pred'),pred) 
