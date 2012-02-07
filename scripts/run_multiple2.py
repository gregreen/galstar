#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       untitled.py
#       
#       Copyright 2011 Gregory <greg@greg-G53JW>
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


import sys, subprocess, shlex


galstar_dir = '/home/greg/projects/galstar/build'
scripts_dir = '/home/greg/projects/galstar/scripts'
output_dir = '/home/greg/projects/galstar/output'

def main():
	infile = str(sys.argv[-2])
	outfile = str(sys.argv[-1])
	
	f = open(infile, 'r')
	for i,line in enumerate(f):
		l = line.rstrip()
		if l:
			out = str(output_dir + "/" + str(outfile + '_' + l.rstrip().replace(' ','_')))
			# Run galstar
			print '====================================================='
			print "%d. RUNNING GALSTAR\n" % (i+1)
			cmd = '%s/galstar %s:DM,Ar --test' % (galstar_dir, out)
			args = shlex.split(cmd)
			p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=galstar_dir)
			stdout, stderr = p.communicate(l)
			print stderr
			print stdout
			# Generate pngs of the pdfs
			print "%d. OUTPUTTING PDFS\n" % (i+1)
			cmd = '%s/plotpdf.py %s DM Ar %s_DM_Ar.png' % (scripts_dir, out, out)
			args = shlex.split(cmd)
			p = subprocess.Popen(args, cwd=galstar_dir)
			p.wait()
	
	return 0

if __name__ == '__main__':
	main()

