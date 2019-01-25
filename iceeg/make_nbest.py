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

def exclude_done(lattice_filenames,output_dir):
	output = []
	ofn = [f.split('/')[-1].split('.')[0] for f in glob.glob(output_dir+'*.nbest')]
	total_lattice = len(lattice_filenames)
	for f in lattice_filenames:
		if f.split('.')[1] not in ofn: output.append(f)
	undone_lattice = len(output)
	print('to do:',undone_lattice,'lattices','total:',total_lattice, 'done:',total_lattice-undone_lattice)
	return output


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
	'''create nbest list for all files in the lattice dir

	lattice_dir 	source dir with lattice files (kaldi output)
	output_dir 		goal dir, to save nbest lists
	nbest 			length of nbest list
	ovewrite 		whether to overwrite previous files (if you want other length nbest list)
	'''
	if output_dir == None: output_dir = '/vol/tensusers/mbentum/PMN_AUDIO/NBEST/'
	fn = get_lattice_filenames(lattice_dir)
	fn = exclude_done(fn, output_dir)
	bar = pb.ProgressBar()
	bar(range(len(fn)))
	for i,f in enumerate(fn):
		bar.update(i)
		make_nbest(f,output_dir,nbest,overwrite)
