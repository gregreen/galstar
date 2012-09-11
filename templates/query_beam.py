#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#  query_beam.py
#  
#  Copyright 2012 Greg Green <greg@greg-UX31A>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


try:
	import lsd
	from lsd.builtins import SFD
except:
	print 'lsd not present.'

import os, sys, argparse
import numpy as np
import pyfits


def mapper(qresult):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
	
	if (obj != None) and (len(obj) > 0):
		yield (1, obj)

def reducer(keyvalue):
	gc_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	yield (gc_index, obj)


def main():
	parser = argparse.ArgumentParser(prog='query_beam.py',
	                                 description='Query beam on sky in lsd',
	                                 add_help=True)
	parser.add_argument('lb', type=float, nargs=2, help='Galactic l and b')
	parser.add_argument('r', type=float, help='radius of beam (in degrees)')
	parser.add_argument('-o', '--output', type=str, required=True,
	                                 help='Output (FITS) filename.')
	parser.add_argument('-s', '--surveys', type=str, nargs='+', default=['ps1'],
	                    choices=('ps1','sdss'), help='Surveys to include')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	args = parser.parse_args(sys.argv[offset:])
	
	print args.surveys
	
	bounds = lsd.bounds.beam(args.lb[0], args.lb[1], radius=args.r, coordsys='gal')
	bounds = lsd.bounds.make_canonical(bounds)
	
	db = lsd.DB(os.environ['LSD_DB'])
	query = None
	if len(args.surveys) == 1:
		if args.surveys[0] == 'ps1':
			query = 'select obj_id, equgal(ra, dec) as (l, b), mean, err, mean_ap, nmag_ok, SFD.EBV(l, b) as EBV from ucal_magsqv where (numpy.sum(nmag_ok > 0, axis=1) >= 4) & (numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2)'
		elif args.surveys[0] == 'sdss':
			print 'Querying from SDSS not implemented yet.'
			return 0
	else:
		print 'Querying from multiple surveys not implemented yet.'
		return 0
	query = db.query(query)
	
	out = []
	for (key, obj) in query.execute([mapper, reducer], 
	                                      group_by_static_cell=True, bounds=bounds):
		tmp = np.empty(len(obj), dtype=[('l', 'f4'), ('b', 'f4'),
		                                ('g', 'f4'), ('g_err', 'f4'),
		                                ('r', 'f4'), ('r_err', 'f4'),
		                                ('i', 'f4'), ('i_err', 'f4'),
		                                ('z', 'f4'), ('z_err', 'f4'),
		                                ('y', 'f4'), ('y_err', 'f4'),
						('g_nmag_ok', 'i4'),
		                                ('r_nmag_ok', 'i4'),
		                                ('i_nmag_ok', 'i4'),
		                                ('z_nmag_ok', 'i4'),
		                                ('y_nmag_ok', 'i4'),
		                                ('EBV', 'f4')])
		tmp['l'] = obj['l']
		tmp['b'] = obj['b']
		tmp['g'] = obj['mean'][:,0]
		tmp['r'] = obj['mean'][:,1]
		tmp['i'] = obj['mean'][:,2]
		tmp['z'] = obj['mean'][:,3]
		tmp['y'] = obj['mean'][:,4]
		tmp['g_err'] = obj['err'][:,0]
		tmp['r_err'] = obj['err'][:,1]
		tmp['i_err'] = obj['err'][:,2]
		tmp['z_err'] = obj['err'][:,3]
		tmp['y_err'] = obj['err'][:,4]
		tmp['g_nmag_ok'] = obj['nmag_ok'][:,0]
		tmp['r_nmag_ok'] = obj['nmag_ok'][:,1]
		tmp['i_nmag_ok'] = obj['nmag_ok'][:,2]
		tmp['z_nmag_ok'] = obj['nmag_ok'][:,3]
		tmp['y_nmag_ok'] = obj['nmag_ok'][:,4]
		tmp['EBV'] = obj['EBV']
		
		out.append(tmp)
	
	out = np.hstack(out)
	print 'max. E(B-V): %.3f' % np.max(out['EBV'])
	pyfits.writeto(args.output, out, clobber=True)
	
	return 0

if __name__ == '__main__':
	main()

