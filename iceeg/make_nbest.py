import glob
import os
import progressbar as pb

bash_script_template = '/vol/tensusers/mbentum/PMN_AUDIO/targz2nbest.template'
bash_script_temp = '/vol/tensusers/mbentum/PMN_AUDIO/targz2nbest.template'

def get_lattice_filenames(lattice_dir = None):
	if lattice_dir == None:
		lattice_dir = '/vol/tensusers/ltenbosch/KALDI_NL/working-directory2/Martijn/'
	fn = glob.glob(lattice_dir + '*.gz')
	fn = [f for f in fn if 'xxx' not in f]
	return fn 



def make_bashscript(f,output_dir = None,nbest = 1000):
	if output_dir == None: output_dir = '/vol/tensusers/mbentum/PMN_AUDIO/NBEST/'
	bs = open(bash_script_template).read()
	name = f.split('.')[1]
	out_f = output_dir + name + '.nbest'
	preamble = '#/bin/bash\n'
	preamble += 'inputfile=' + f + '\n'
	preamble += 'outputfile=' + out_f + '\n'
	preamble += 'Nbest=' + str(nbest) + '\n'
	with open('targz2nbest','w') as fout: fout.write(preamble + bs)
	os.system('chmod 775 targz2nbest')
	return out_f

def make_nbest(f,output_dir = None, nbest = 1000, overwrite = False):
	if output_dir == None: output_dir = '/vol/tensusers/mbentum/PMN_AUDIO/NBEST/'
	out_f = make_bashscript(f,output_dir,nbest)
	if os.path.isfile(out_f):print(f,'already exists, doing nothing')
	else:
		print('generating nbest, saving:',out_f)
		os.system('sh targz2nbest')

def make_all_nbest(lattice_dir = None,output_dir = None, nbest = 1000, overwrite=False):
	fn = get_lattice_filenames(lattice_dir)
	bar = pb.ProgressBar()
	bar(range(len(fn)))
	for i,f in enumerate(fn):
		bar.update(i)
		make_nbest(f,output_dir,nbest,overwrite)
