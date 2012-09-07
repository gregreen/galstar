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

import lsd, os
from iterators import block_string_by_comments, index_by_key


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
	
	'''
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
	'''
	
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

def cartesian(arrays, out=None):
	"""
	Generate a cartesian product of input arrays.
	
	Parameters
	----------
	arrays : list of array-like
	    1-D arrays to form the cartesian product of.
	out : ndarray
	    Array to place the cartesian product in.
	
	Returns
	-------
	out : ndarray
	    2-D array of shape (M, len(arrays)) containing cartesian products
	    formed of input arrays.
	
	Examples
	--------
	>>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
	array([[1, 4, 6],
	       [1, 4, 7],
	       [1, 5, 6],
	       [1, 5, 7],
	       [2, 4, 6],
	       [2, 4, 7],
	       [2, 5, 6],
	       [2, 5, 7],
	       [3, 4, 6],
	       [3, 4, 7],
	       [3, 5, 6],
	       [3, 5, 7]])
	
	"""
	
	arrays = [np.asarray(x) for x in arrays]
	dtype = arrays[0].dtype
	
	n = np.prod([x.size for x in arrays])
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=dtype)
	
	m = n / arrays[0].size
	out[:,0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m,1:])
		for j in xrange(1, arrays[0].size):
			out[j*m:(j+1)*m,1:] = out[0:m,1:]
	return out

def great_circle_dist(tp0, tp1):
	'''
	Returns the great-circle distance bewteen two sets of coordinates,
	tp0 and tp1.
	
	Inputs:
	    tp0  (N, 2) numpy array. Each element is (theta, phi) in rad.
	    tp1  (M, 2) numpy array. Each element is (theta, phi) in rad.
	
	Output:
	    dist  (N, M) numpy array. dist[n,m] = dist(tp0[n], tp1[m]).
	'''
	
	N = tp0.shape[0]
	M = tp1.shape[0]
	out = np.empty((N,M), dtype=tp0.dtype)
	
	dist = lambda p0, t0, p1, t1: np.arccos(np.sin(t0)*np.sin(t1)
	                              + np.cos(t0)*np.cos(t1)*np.cos(p0-p1))
	
	if N <= M:
		for n in xrange(N):
			out[n,:] = dist(tp0[n,1], tp0[n,0], tp1[:,1], tp1[:,0])
	else:
		for m in xrange(M):
			out[:,m] = dist(tp0[:,1], tp0[:,0], tp1[m,1], tp1[m,0])
	
	return out


def mapper(qresult, index, pos):
	obj = lsd.colgroup.fromiter(qresult, blocks=True)
		
	if (obj != None) and (len(obj) > 0):
		# Find nearest cluster center to each star
		theta_star = (90. - obj['b']) * np.pi/180.
		phi_star = obj['l'] * np.pi/180.
		tp_star = np.array([theta_star, phi_star]).T
		tp_gc = pos[:2].T
		d = great_circle_dist(tp_star, tp_gc)
		min_idx = np.argmin(d, axis=1)
		#d_min = np.min(d, axis=1)
		
		for gc_idx,block_idx in index_by_key(min_idx):
			yield (gc_idx, obj[block_idx])

def reducer(keyvalue):
	gc_index, obj = keyvalue
	obj = lsd.colgroup.fromiter(obj, blocks=True)
	
	yield (gc_index, obj)

def query_gcs(data, outfname):
	# Determine bounds for query
	print 'Constructing bounds...'
	q_bounds = []
	idx = np.empty(len(data), dtype=np.bool)
	idx.fill(False)
	index, l, b, r_gc = [], [], [], []
	for i,d in enumerate(data):
		r = 15. * d['r_h'] / 60.
		if np.isnan(r):
			r = 20. * d['r_c'] / 60.
		if not np.isnan(r):
			idx[i] = True
			index.append(i)
			l.append(d['l'])
			b.append(d['b'])
			r_gc.append(r)
			q_bounds.append(lsd.bounds.beam(d['l'], d['b'], radius=r,
			                                           coordsys='gal'))
	q_bounds = lsd.bounds.make_canonical(q_bounds)
	pos = np.array([l, b, r_gc])
	pos[1,:] = 90. - pos[1,:]
	pos *= np.pi/180.
	index = np.array(index)
	
	# Set up query
	print 'Setting up bounds...'
	db = lsd.DB(os.environ['LSD_DB'])
	query = "select obj_id, equgal(ra, dec) as (l, b), mean, err, mean_ap, nmag_ok from ucal_magsqv where (numpy.sum(nmag_ok > 0, axis=1) >= 4) & (numpy.sum(mean - mean_ap < 0.1, axis=1) >= 2)"
	query = db.query(query)
	
	# Execute query
	print 'Executing query...'
	out = []
	for (i, obj) in query.execute([(mapper, index, pos), reducer], 
	                                      group_by_static_cell=True, bounds=q_bounds):
		tmp = np.empty(len(obj), dtype=[('gcID','S12'),
		                                 ('l','f4'), ('b','f4'),
		                                 ('g','f4'), ('g_err','f4'),
		                                 ('r','f4'), ('r_err','f4'),
		                                 ('i','f4'), ('i_err','f4'),
		                                 ('z','f4'), ('z_err','f4'),
		                                 ('y','f4'), ('y_err','f4')])
		tmp['gcID'][:] = data['ID'][i]
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
		
		out.append(tmp)
	
	print 'Writing output to %s ...' % outfname
	out = np.hstack(out)
	pyfits.writeto(outfname, out, clobber=True)


def main():
	data = gc_txt2fits('mwgc')
	query_gcs(data, 'gcstars.fits')
	
	return 0

if __name__ == '__main__':
	main()

