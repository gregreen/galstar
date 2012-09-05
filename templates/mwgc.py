#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#  untitled.py
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

import numpy as np
import pyfits
from cStringIO import StringIO
from numpy.lib.recfunctions import join_by

from iterators import block_string_by_comments


def gc_txt2fits(fname_base):
	f = open('%s.txt' % fname_base, 'r')
	txt = f.read()
	f.close()
	
	col_fmt = []
	col_len = []
	
	col_fmt.append([('ID', 'S11'), ('Name', 'S12'),
	                ('RAh', 'i1'), ('RAm', 'i1'), ('RAs', 'f4'),
	                ('DECh', 'i1'), ('DECm', 'i1'), ('DECs', 'f4'),
	                ('l', 'f4'), ('b', 'f4'),
	                ('R_Sun', 'f4'), ('R_gc', 'f4'),
	                ('X', 'f4'), ('Y', 'f4'), ('Z', 'f4')])
	
	col_fmt.append([('ID', 'S11'),
	                ('FeH', 'f4'), ('FeH_wt', 'i2'),
	                ('EBV', 'f4'), ('VHB', 'f4'),
	                ('DM_V', 'f4'), ('V_t', 'f4'), ('M_Vt', 'f4'),
	                ('UB', 'f4'), ('BV', 'f4'),
	                ('VR', 'f4'), ('RI', 'f4'),
	                ('spt', 'S5'),
	                ('ellip', 'f4')])
	
	col_fmt.append([('ID', 'S11'),
	                ('v_r', 'f4'), ('v_r_err', 'f4'),
	                ('v_LSR', 'f4'),
	                ('sig_v', 'f4'), ('sig_v_err', 'f4'),
	                ('c', 'f4'), ('r_c', 'f4'), ('r_h', 'f4'),
	                ('mu_V', 'f4'), ('rho_0', 'f4'),
	                ('log_tc', 'f4'), ('log_th', 'f4')])
	
	comment_unit = {('ID': 'GLOBULAR CLUSTER ID'),
	                ('Name': 'NAME OF GC'),
	                ('RAh': 'RA HOUR'),
	                ('RAm': 'RA MINUTE'),
	                ('RAs': 'RA SECOND'),
	                ('DECh': 'DEC HOUR'),
	                ('DECm': 'DEC MINUTE'),
	                ('DECs': 'DEC SECOND'),
	                ('l': 'GALACTIC LONGITUDE'),
	                ('b': 'GALACTIC LATITUDE'),
	                ('R_Sun': 'DIST FROM SUN'),
	                ('R_gc': 'DIST FROM GALACTIC CENTER'),
	                ('X': 'CARTESIAN X DISP FROM GAL CENTER'),
	                ('Y': 'CARTESIAN Y DISP FROM GAL CENTER'),
	                ('Z': 'CARTESIAN Z DISP FROM GAL CENTER'),
	                ('FeH': 'METALLICITY'),
	                ('FeH_wt': 'WEIGHT OF FEH MEASUREMENT'),
	                ('EBV': 'B-V EXCESS'),
	                ('VHB': ''),
	                ('DMV': 'DIST MODULUS FROM V BAND'),
	                ('V_t': ''),
	                ('M_Vt': ''),
	                ('UB': 'U-B COLOR'),
	                ('BV': 'B-V COLOR'),
	                ('VR': 'V-R COLOR'),
	                ('RI': 'R-I COLOR'),
	                ('spt': 'INTEGRATED SPECTRAL TYPE'),
	                ('ellip': ''),
	                ('v_r': 'HELIOCENTRIC RADIAL VELOCITY'),
	                ('v_r_err': 'UNCERTAINTY IN v_r'),
	                ('v_LSR': 'RAD VEL RELATIVE TO LSR'),
	                ('sig_v': 'CENTRAL VELOCITY DISP'),
	                ('sig_v_err': 'UNCERTAINTY IN sig_v_err'),
	                ('c': 'CONCENTRATION PARAMETER'),
	                ('r_c': 'RADIUS OF CORE'),
	                ('r_h': 'HALF-LIGHT RADIUS'),
	                ('mu_V': 'V-BAND SURFACE BRIGHTNESS'),
	                ('rho_0': 'SURFACE NUMBER DENSITY'),
	                ('log_tc': 'CORE RELAXATION TIME'),
	                ('log_th': 'MEDIAN RELAXATION TIME')}
	
	col_len.append([11, 13, 3, 3, 7, 4, 3, 7, 8, 8, 6, 6, 6, 6, 5])
	col_len.append([11, 7, 5, 5, 6, 6, 6, 7, 7, 6, 6, 6, 6, 5])
	col_len.append([11, 8, 6, 8, 8, 7, 8, 8, 8, 7, 7, 7, 5])
	
	formatted_txt = []
	for i,s in enumerate(block_string_by_comments(txt)):
		rows = []
		
		for line in s.splitlines():
			# Ignore comments and blank lines
			line = line.lstrip()
			if len(line) == 0:
				continue
			elif line[0] == '#':
				continue
			
			# Read in columns of constant width
			cols = []
			start = 0
			ncols = 0
			for c in col_len[i]:
				if start + c > len(line):
					break
				tmp = line[start:start+c].lstrip().rstrip()
				if tmp == '':
					tmp = 'NaN'
				cols.append(tmp)
				ncols += 1
				start += c
			
			# Fill in missing columns at end
			for k in xrange(ncols, len(col_len[i])):
				cols.append('NaN')
			
			# Join columns, using tabs as delimiters
			rows.append('\t'.join(cols))
		
		# Join rows, using endlines as delimiters
		formatted_txt.append('\n'.join(rows))
	
	# Convert formatted strings into numpy record arrays
	d = []
	for fmt,s in zip(col_fmt, formatted_txt):
		d.append(np.genfromtxt(StringIO(s), dtype=fmt, delimiter='\t'))
	
	
	# Merge record arrays by name
	out = join_by('ID', d[0], d[1], jointype='outer')
	out = join_by('ID', out, d[2], jointype='outer')
	out['Name'][out['Name'] == 'NaN'] = ''
	out['spt'][out['spt'] == 'NaN'] = ''
	
	# Output record array to FITS file
	'''
	cols = []
	cols.append(pyfits.Column(name='MU', format='%dD' % len(mu), array=mu))
	for i, m in enumerate(maps):
		cols.append(pyfits.Column(name='A_R %d' % i, format='D', array=m))
	
	tbhdu = pyfits.new_table(cols)
	tbhdu.header.update('NESTED', nest, 'Healpix ordering scheme.')
	tbhdu.header.update('NSIDE', hp.npix2nside(maps.shape[1]), 'Healpix nside parameter.')
	
	tbhdu.writeto(fname, clobber=True)
	'''
	
	#hdu = []
	#hdu.append(pyfits.PrimaryHDU(mu))
	#for m in maps:
	#	hdu.append(pyfits.ImageHDU(m))
	#hdulist = pyfits.HDUList(hdu)
	#hdulist.writeto(fname, clobber=True)
	
	pyfits.writeto('%s.fits' % fname_base, out, clobber=True)
	
	return out



def main():
	data = gc_txt2fits('mwgc')
	
	return 0

if __name__ == '__main__':
	main()

