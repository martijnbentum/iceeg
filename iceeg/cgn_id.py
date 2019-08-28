import glob
from lxml import etree
import os
import path

def get_dtd():
	return etree.DTD(path.cgn_xml + 'ttext.dtd')


def get_fn(directory):
	if not directory.endswith('/'): directory += '/'
	return glob.glob(directory+'fn*.skp')


def fix_diacritics(text,dtd = None):
	if dtd == None: dtd = get_dtd()
	for o in dtd.iterentities():
		text = text.replace('&'+o.name+';',o.content)
	return text


def make_xml_files(source_dir,goal_dir, make = False, overwrite = False):
	if not source_dir.endswith('/'): source_dir+= '/'
	if not goal_dir.endswith('/'): goal_dir+= '/'
	if not os.path.isdir(goal_dir) and make: os.mkdir(goal_dir)
	fn = get_fn(source_dir)
	print('found:',len(fn),'files',source_dir)
	for f in fn:
		t = open(f).read()
		t = fix_diacritics(t)
		name = goal_dir + f.split('/')[-1].split('.')[0] + '.xml'
		if not make:
			print(f,'>>',name,'test, did nothing')
			continue
		if os.path.isfile(name) and not overwrite: continue
		with open(name,'w') as fout:
			fout.write(t)


def get_comps(directory):
	if not directory.endswith('/'): directory += '/'
	return [d + '/nl/' for d in glob.glob(directory+'comp-*')]


def make_all_ort(make = False,overwrite = False):
	d = path.cgn_xml + 'skp-ort/'
	g = path.cgn_xml + 'ort_xml/'
	comps = get_comps(d)
	for comp in comps:
		print('working on:',comp)
		make_xml_files(comp,g, make=make,overwrite=overwrite)
				
	
	
