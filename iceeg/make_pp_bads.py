import combine_artifacts

def make_pp_bads(exp,fo):
	bads = ca.load_all_bads(exp,fo)
	for bad in bads:
		if int(bad.name.split('_')[0][2:]) in pp.keys():
			pp[int(bad.name.split('_')[0][2:])].append(bad)
		else:
			pp[int(bad.name.split('_')[0][2:])]=[bad]
	return pp, bads
