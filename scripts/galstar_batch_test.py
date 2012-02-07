#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       galstar_batch_test.py
#       
#       Copyright 2011 Greg <greg@greg-G53JW>
#       
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.
#       
#       

import sys
import argparse
import subprocess
import shlex
from os.path import abspath
from random import random

galstar_bin = '/home/greg/projects/galstar/build/galstar'

def main():
	parser = argparse.ArgumentParser(prog='galstar_batch_tset.py', description='Run the galstar --test option multiple times with random stellar parameters', add_help=True)
	parser.add_argument('fname', type=str, help='Output filename base (without extension)')
	parser.add_argument('N', type=int, help='# of stars to generate')
	parser.add_argument('--lb', type=float, nargs=2, help='Galactic l and b of stars (default: random star positions)')
	parser.add_argument('--errors', type=float, nargs=5, default=(0.2, 0.1, 0.1, 0.1, 0.1), help='ugriz errors')
	parser.add_argument('--steps', type=int, default=15000, help='# of steps per sampler in galstar')
	if sys.argv[0] == 'python':
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	print 'Running galstar --test %d times ...\n' % values.N
	
	params_list = ''
	
	for i in range(values.N):
		# Determine filename
		fn = abspath(values.fname + '_%d' % i)
		
		# Generate galstar --test input
		if values.lb != None:
			l,b = values.lb
		else:
			l,b = random()*180., random()*90.
		DM = random()*14.+5.
		Ar = random()*5.
		Mr = random()*29.-1.
		FeH = -random()*2.5
		
		params_list += '%.3f %.3f %.3f %.3f\n' % (DM, Ar, Mr, FeH)
		
		# Generate command and feed stellar parameters to galstar --test
		cmd = 'echo "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" | %s %s:DM,Ar --test --statsfile %s --steps %d' % (l, b, Ar, DM, Mr, FeH, values.errors[0], values.errors[1], values.errors[2], values.errors[3], values.errors[4], galstar_bin, fn, fn, values.steps)
		print '\n' + cmd + '\n'
		proc = subprocess.Popen(cmd, shell=True)
		proc.wait()
	
	# Output true stellar parameters to .truth file
	print 'Writing summary of true stellar parameters to %s ...' % (values.fname + '.truth')
	f = open(abspath(values.fname + '.truth'), 'w')
	f.write(params_list)
	f.close()
	
	return 0

if __name__ == '__main__':
	main()

