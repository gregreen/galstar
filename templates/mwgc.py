#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
#
#  mwgc.py
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
#  Globular clusters in catalog:
#
#  NGC 6656 514698
#  NGC 6626 198863
#  NGC 6638 173330
#  NGC 6121 146067
#  Terzan 12 137798
#  IC 1276 119375
#  NGC 6144 101167
#  NGC 6712 96580
#  NGC 6838 96009
#  NGC 6304 91225
#  NGC 6273 89835
#  NGC 6760 71536
#  NGC 6544 69658
#  NGC 6539 59914
#  NGC 6366 59630
#  NGC 6642 59174
#  Terzan 1 55627
#  NGC 6355 52238
#  Pal 6 50131
#  NGC 6553 48880
#  NGC 6401 48257
#  Terzan 4 42227
#  NGC 6293 42225
#  Pal 10 39239
#  NGC 6254 32350
#  NGC 6333 31855
#  NGC 6624 30525
#  NGC 6316 28524
#  NGC 6356 26279
#  NGC 6218 25638
#  Pal 11 25615
#  NGC 6749 24690
#  NGC 6205 24429
#  NGC 6235 23837
#  Terzan 3 23578
#  Terzan 8 22469
#  NGC 6171 21563
#  NGC 6342 21400
#  Arp 2 21162
#  BH 261 20881
#  NGC 6402 19507
#  NGC 6284 19124
#  NGC 6779 18310
#  IC 1257 17648
#  NGC 7078 16666
#  NGC 6266 16620
#  Pal 8 16464
#  NGC 6717 16373
#  NGC 5897 15997
#  NGC 6440 15773
#  Djorg 2 14799
#  HP 1 14272
#  NGC 6325 13687
#  NGC 5272 13049
#  NGC 6535 12622
#  Rup 106 12285
#  NGC 6287 10778
#  NGC 7089 9759
#  NGC 6341 9431
#  NGC 5904 8543
#  NGC 6426 8322
#  NGC 5466 8175
#  NGC 7099 7895
#  Pal 5 7746
#  NGC 6540 7078
#  NGC 6809 6809
#  NGC 288 5993
#  Pal 15 5353
#  1636-283 5322
#  GLIMPSE01 5316
#  NGC 6934 5274
#  NGC 6715 5213
#  NGC 4590 5192
#  NGC 6981 4835
#  Pal 12 4691
#  GLIMPSE02 3365
#  Ton 2 3284
#  NGC 1904 3204
#  Terzan 10 2926
#  NGC 6864 2739
#  Pal 14 2589
#  NGC 6517 2281
#  NGC 6093 2192
#  NGC 2419 1941
#  NGC 7006 1550
#  2MS-GC02 1363
#  NGC 5634 1253
#  Pal 2 1093
#  NGC 7492 933
#  NGC 6229 859
#  NGC 5694 783
#  NGC 4147 763
#  2MS-GC01 673
#  Pal 1 492
#  NGC 5053 430
#  Ko 1 259
#  Pal 13 204
#  Eridanus 169
#  Ko 2 80
#  Pal 3 68
#  AM 4 62
#  UKS 1 42
#  Pal 4 39
#  NGC 362 0
#  NGC 3201 0
#  NGC 2808 0
#  AM 1 0
#  Lynga 7 0
#  NGC 2298 0
#  NGC 1851 0
#  BH 176 0
#  Liller 1 0
#  NGC 1261 0
#  Djorg 1 0
#  E 3 0
#  ESO-SC06 0
#  FSR 1735 0
#  IC 4499 0
#  NGC 104 0
#  NGC 4372 0
#  Whiting 1 0
#  NGC 4833 0
#  NGC 5024 0
#  NGC 6541 0
#  NGC 6558 0
#  NGC 6569 0
#  NGC 6584 0
#  NGC 6637 0
#  NGC 6652 0
#  NGC 6681 0
#  NGC 6723 0
#  NGC 6752 0
#  Pyxis 0
#  Terzan 2 0
#  Terzan 5 0
#  Terzan 6 0
#  Terzan 7 0
#  Terzan 9 0
#  NGC 6528 0
#  NGC 6522 0
#  NGC 6496 0
#  NGC 6101 0
#  NGC 5139 0
#  NGC 5286 0
#  NGC 5824 0
#  NGC 5927 0
#  NGC 5946 0
#  NGC 5986 0
#  NGC 6139 0
#  NGC 6453 0
#  NGC 6256 0
#  NGC 6352 0
#  NGC 6380 0
#  NGC 6388 0
#  NGC 6397 0
#  NGC 6441 0
#  NGC 6362 0
#

import numpy as np
import pyfits
import os
from cStringIO import StringIO
from numpy.lib.recfunctions import join_by
from iterators import block_string_by_comments, index_by_key, index_by_unsortable_key

try:
	import lsd
except:
	print 'lsd not present'

import matplotlib.pyplot as plt
import matplotlib as mplib
from plot_sdss_photometry import density_scatter

from stellarmodel import StellarModel


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
	
	try:
		pyfits.writeto('%s.fits' % fname_base, out, clobber=False)
	except IOError, e:
		print e
	
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
	pos = np.array([b, l, r_gc])
	pos[0,:] = 90. - pos[0,:]
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


def plot_gc(gcstars, gcdata, gcID, cut=5.0, err=0.5, modelfn='../data/PScolors.dat'):
	d = gcstars[gcstars['gcID'] == gcID]
	gc = gcdata[gcdata['ID'] == gcID]
	
	# Filter out stars which are too far from cluster center
	gc_rad = gc['r_h'] / 60.
	if np.isnan(gc_rad):
		gc_rad = gc['r_c'] / 60.
	tp_star = np.array([(90. - d['b'])*np.pi/180., d['l']*np.pi/180.]).T
	tp_gc = np.array([(90. - gc['b'])*np.pi/180., gc['l']*np.pi/180.]).T
	gc_dist = great_circle_dist(tp_gc, tp_star)[0]
	#idx = np.argsort(gc_dist)
	#d = d[idx[:int(cut*len(idx))]]
	idx = (gc_dist < cut * gc_rad * np.pi / 180.)
	d = d[idx]
	print 'Filtered out %.1f pct of stars based on distance.' % (100.*(1. - float(np.sum(idx))/float(len(idx))))
	
	# Determine distance modulus to cluster
	mu = 5. * (2. + np.log10(gc['R_Sun'][0]))
	
	print 'Cluster ID: %s' % gcID
	print 'Radius = %.2f deg' % gc_rad
	print '# of stars = %d' % len(d)
	print 'DM = %.2f' % mu
	print '[Fe/H] = %.2f' % gc['FeH'] 
	print 'E(B-V) = %.2f' % gc['EBV']
	
	# Correct observed magnitudes for extinction
	A_coeff = np.array([3.172, 2.271, 1.682, 1.322, 1.087])
	EBV = gc['EBV'] + 0.15
	m_g = d['g'] - A_coeff[0] * EBV
	m_r = d['r'] - A_coeff[1] * EBV
	m_i = d['i'] - A_coeff[2] * EBV
	m_z = d['z'] - A_coeff[3] * EBV
	m_y = d['y'] - A_coeff[4] * EBV
	
	# Determine model colors
	model = StellarModel(modelfn)
	idx = (d['g'] > 0.) & (d['r'] > 0.)
	idx = idx & (d['g_err'] < 10.*err) & (d['r_err'] < 10.*err)
	Mr = np.linspace(max(np.min(m_r[idx] - mu), -1.), min(np.max(m_r[idx] - mu), 27.), 200)
	FeH = np.empty(200, dtype=np.float32)
	FeH.fill(gc['FeH'])
	model_color = []
	for i in range(3):
		model_color.append(model.color(Mr, FeH + 0.5*(i-1.)))
	
	mplib.rc('text', usetex=True)
	
	'''
	# Color - Color
	fig = plt.figure(figsize=(8,8), dpi=100)
	
	# g-r vs r-i
	ax = fig.add_subplot(2,2,2)
	idx = (d['g'] > 0.) & (d['r'] > 0.) & (d['i'] > 0.)
	idx = idx & (d['g_err'] < err) & (d['r_err'] < err) & (d['i_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_g[idx] - m_r[idx], m_r[idx] - m_i[idx],
		                binsize=(0.05,0.05), threshold=8)
		ax.set_xlabel(r'$g - r$', fontsize=16)
		ax.set_ylabel(r'$r - i$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['gr'], mc['ri'], c, alpha=0.5)
	
	# r-i vs i-z
	ax = fig.add_subplot(2,2,3)
	idx = (d['r'] > 0.) & (d['i'] > 0.) & (d['z'] > 0.)
	idx = idx & (d['r_err'] < err) & (d['i_err'] < err) & (d['z_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_r[idx] - m_i[idx], m_i[idx] - m_z[idx],
		                binsize=(0.05,0.05), threshold=8)
		ax.set_xlabel(r'$r - i', fontsize=16)
		ax.set_ylabel(r'$i - z$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['ri'], mc['iz'], c, alpha=0.5)
	
	# i-z vs z-y
	ax = fig.add_subplot(2,2,4)
	#for i in range(50):
	#	print d['i'][5*i:5*(i+1)]
	#	print d['i_err'][5*i:5*(i+1)]
	#	print ''
	idx = (d['i'] > 0.) & (d['z'] > 0.) & (d['y'] > 0.)
	idx = idx & (d['i_err'] < err) & (d['z_err'] < err) & (d['y_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_i[idx] - m_z[idx], m_z[idx] - m_y[idx],
		                binsize=(0.05,0.05), threshold=8)
		ax.set_xlabel(r'$i - z$', fontsize=16)
		ax.set_ylabel(r'$z - y$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['iz'], mc['zy'], c, alpha=0.5)
	'''
	
	
	# Color - Magnitude
	fig = plt.figure(figsize=(8,8), dpi=100)
	
	# g-r vs M_r
	ax = fig.add_subplot(2,2,1)
	idx = (m_g > 0.) & (m_r > 0.)
	idx = idx & (d['g_err'] < err) & (d['r_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_g[idx] - m_r[idx], m_r[idx] - mu,
		                nbins=(500,500), threshold=100)
		ax.set_xlim(-0.5, 0.9)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		ax.set_xlabel(r'$g - r$', fontsize=16)
		ax.set_ylabel(r'$M_{r}$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['gr'], Mr, c, alpha=0.5)
	
	# r-i vs M_r
	ax = fig.add_subplot(2,2,2)
	idx = (m_r > 0.) & (m_i > 0.)
	idx = idx & (d['r_err'] < err) & (d['i_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_r[idx] - m_i[idx], m_r[idx] - mu,
		                nbins=(500,500), threshold=100)
		ax.set_xlim(-0.5, 0.7)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		ax.set_xlabel(r'$r - i$', fontsize=16)
		ax.set_ylabel(r'$M_{r}$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['ri'], Mr, c, alpha=0.5)
	
	# i-z vs M_r
	ax = fig.add_subplot(2,2,3)
	idx = (m_r > 0.) & (m_i > 0.) & (m_z > 0.)
	idx = idx & (d['i_err'] < err) & (d['z_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_i[idx] - m_z[idx], m_r[idx] - mu,
		                nbins=(500,500), threshold=100)
		ax.set_xlim(-0.5, 0.6)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		ax.set_xlabel(r'$i - z$', fontsize=16)
		ax.set_ylabel(r'$M_{r}$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['iz'], Mr, c, alpha=0.5)
	
	# z-y vs M_r
	ax = fig.add_subplot(2,2,4)
	idx = (m_r > 0.) & (m_z > 0.) & (m_y > 0.)
	idx = idx & (d['z_err'] < err) & (d['y_err'] < err)
	if np.any(idx):
		density_scatter(ax, m_z[idx] - m_y[idx], m_r[idx] - mu,
		                nbins=(500,500), threshold=100)
		ax.set_xlim(-0.5, 0.6)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[1], ylim[0])
		ax.set_xlabel(r'$z - y$', fontsize=16)
		ax.set_ylabel(r'$M_{r}$', fontsize=16)
		for c,mc in zip(['r','g','c'], model_color):
			ax.plot(mc['zy'], Mr, c, alpha=0.5)
	
	
	# Source positions
	fig = plt.figure(figsize=(8,8), dpi=100)
	ax = fig.add_subplot(1,1,1)
	density_scatter(ax, d['l'], d['b'], nbins=(100,100), threshold=20)
	ax.set_xlabel(r'$\ell$', fontsize=16)
	ax.set_ylabel(r'$b$', fontsize=16)


def print_summary(gcdata, gcstars):
	NStars = np.empty(gcdata.size, dtype=np.uint32)
	for i,gcID in enumerate(gcdata['ID']):
		idx = gcstars['gcID'] == gcID
		NStars[i] = np.sum(idx)
		#print gcID, np.sum(idx)
	
	idx = np.argsort(NStars)[::-1]
	print 'Cluster ID:\t# of Stars'
	for i in idx:
		print gcdata['ID'][i], NStars[i]
	
	return NStars


def gen_input(data, fname):
	# Write header
	f = open(fname, 'wb')
	f.write(np.array(len(data), dtype=np.uint32).tostring())
	
	for i,d in enumerate(data):
		# Filter data based on number of detections
		count = np.zeros(len(d), dtype='i4')
		for b in ['g', 'r', 'i', 'z', 'y']:
			count += (d[b] > 0.).astype('i4')
		idx = (count >= 4) & (d['g'] > 0.)
		d = d[idx]
		
		# Inflate errors of nondetections
		for b in ['g', 'r', 'i', 'z', 'y']:
			idx = (d[b] <= 0.)
			b_err = '%s_err' % b
			d[b_err][idx] = 1.e10
		
		outarr = np.empty(len(d), dtype=[('obj_id', 'u8'),
		                                 ('l', 'f8'), ('b', 'f8'),
		                                 ('g', 'f8'),
		                                 ('r', 'f8'),
		                                 ('i', 'f8'),
		                                 ('z', 'f8'),
		                                 ('y', 'f8'),
		                                 ('g_err', 'f8'),
		                                 ('r_err', 'f8'),
		                                 ('i_err', 'f8'),
		                                 ('z_err', 'f8'),
		                                 ('y_err', 'f8')])
		outarr['obj_id'] = 0
		for a in ['l', 'b', 'g', 'r', 'i', 'z', 'y',
		          'g_err', 'r_err', 'i_err', 'z_err', 'y_err']:
			outarr[a] = d[a]
		
		print outarr['g']
		
		# Write pixel header
		N_stars = np.array([outarr.shape[0]], dtype=np.uint32)
		gal_lb = np.array([np.mean(outarr['l']), np.mean(outarr['b'])], dtype=np.float64)
		f.write(np.array([i], dtype=np.uint32).tostring())	# Pixel index	(uint32)
		f.write(gal_lb.tostring())							# (l, b)		(2 x float64)
		f.write(N_stars.tostring())							# N_stars		(uint32)
		
		# Write stellar data
		f.write(outarr.tostring())	# obj_id, l, b, 5xmag, 5xerr	(uint64 + 12 x float64)
	
	f.close()


def main():
	gcdata = gc_txt2fits('mwgc')
	gcstars = pyfits.getdata('gcstars.fits')
	
	#query_gcs(gcdata, 'gcstars.fits')
	
	#NStars = print_summary(gcdata, gcstars)
	#idx = np.argsort(NStars)[::-1]
	#gcID = gcdata['ID'][idx[0]]
	MessierGC = ['NGC 7089',
	             'NGC 5272',
	             'NGC 6121',
	             'NGC 5904',
	             'NGC 6333',
	             'NGC 6254',
	             'NGC 6218',
	             'NGC 6205',
	             'NGC 6402',
	             'NGC 7078',
	             'NGC 6273',
	             'NGC 6656',
	             'NGC 6626',
	             'NGC 7099',
	             'NGC 5024',
	             'NGC 6715',
	             'NGC 6809',
	             'NGC 6779',
	             'NGC 6266',
	             'NGC 4590',
	             'NGC 6637',
	             'NGC 6681',
	             'NGC 6838',
	             'NGC 6981',
	             'NGC 6864',
	             'NGC 1904',
	             'NGC 6093',
	             'NGC 6341',
	             'NGC 6171']
	gcID = MessierGC[2]
	plot_gc(gcstars, gcdata, gcID, cut=1., err=0.05, modelfn='../data/PScolors.dat')
	plt.show()
	
	# Generate input files from selected globular clusters
	cut = [1., 2., 2., 2., 4., 5., 4., 3., 6., 5., 3.]
	index = [2, 5, 6, 8, 9, 13, 19, 23, 25, 27, 28]
	
	output_data = []
	
	'''for i,c in zip(index, cut):
		gcID = MessierGC[i]
		
		d = gcstars[gcstars['gcID'] == gcID]
		gc = gcdata[gcdata['ID'] == gcID]
		
		# Filter out stars which are too far from cluster center
		gc_rad = gc['r_h'] / 60.
		if np.isnan(gc_rad):
			gc_rad = gc['r_c'] / 60.
		tp_star = np.array([(90. - d['b'])*np.pi/180., d['l']*np.pi/180.]).T
		tp_gc = np.array([(90. - gc['b'])*np.pi/180., gc['l']*np.pi/180.]).T
		gc_dist = great_circle_dist(tp_gc, tp_star)[0]
		idx = (gc_dist < c * gc_rad * np.pi / 180.)
		
		output_data.append(d[idx])
	
	gen_input(output_data, 'GCs.in')'''
	
	return 0

if __name__ == '__main__':
	main()

