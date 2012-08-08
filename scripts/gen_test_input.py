#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  gen_test_input.py
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

import sys, argparse
import numpy as np

def main():
	parser = argparse.ArgumentParser(prog='gen_test_input.py', description='Generates test input file for galstar.', add_help=True)
	parser.add_argument('coordinates', type=float, nargs=2, metavar='l b', help='Galactic latitude and longitude, in degrees.')
	parser.add_argument('N', type=int, help='# of stars to generate.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	params = np.random.rand(values.N, 4)
	params[:,0] = params[:,0] * 13.5 + 5.5
	params[:,1] = params[:,1] * 5.
	params[:,2] = params[:,2] * 20. - 0.8
	params[:,3] = params[:,3] * 2.4 - 2.45
	
	idx = (params[:,0] + params[:,1] + params[:,2] > 22.)
	
	while np.any(idx):
		params[idx,:3] = np.random.rand(np.sum(idx), 3)
		params[idx,0] = params[idx,0] * 13.5 + 5.5
		params[idx,1] = params[idx,1] * 5.
		params[idx,2] = params[idx,2] * 20. - 0.8
		idx = (params[:,0] + params[:,1] + params[:,2] > 22.)
	
	header = '''# Format:
# l  b
# DM  Ar  Mr  FeH
# DM  Ar  Mr  FeH
# DM  Ar  Mr  FeH
# ...'''
	print header
	print '%.3f  %.3f' % (values.coordinates[0], values.coordinates[1])
	for p in params:
		print '%.3f  %.3f  %.3f  %.3f' % (p[0], p[1], p[2], p[3])
	
	return 0

if __name__ == '__main__':
	main()

