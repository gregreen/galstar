#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       healpix_utils.py
#       
#       Copyright 2012 Greg <greg@greg-G53JW>
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

# TODO: Change to nest=True if query_lsd.py is run again.
# TODO: Add pix_index option to evaluate function

import matplotlib.pyplot as plt
import matplotlib as mplib
import numpy as np
import healpy as hp
import pyfits

from os.path import abspath


def lb2thetaphi(l, b):
	'''
	Convert Galactic (l, b) to spherical coordinates (theta, phi).
	Uses the physics convention, where theta is in the range [0, 180],
	and phi is in the range [0, 360].
	
	Input:
	    l      Galactic longitude, in degrees
	    b      Galactic latitude, in degrees
	
	Output:
	    theta  Angle from the z-axis, in radians
	    phi    Angle from the prime meridian, in radians
	'''
	return np.pi/180. * (90. - b), np.pi/180. * l


def thetaphi2lb(theta, phi):
	'''
	Convert spherical (theta, phi) to Galactic coordinates (l, b).
	Uses the physics convention, where theta is in the range [0, 180],
	and phi is in the range [0, 360].
	
	Input:
	    theta  Angle from the z-axis, in radians
	    phi    Angle from the prime meridian, in radians
	
	Output:
	    l      Galactic longitude, in degrees
	    b      Galactic latitude, in degrees
	'''
	return 180./np.pi * phi, 90. - 180./np.pi * theta


def deg2rad(theta):
	'''
	Convert from degrees to radians.
	
	Input:
	    theta    angle or numpy array of angles, in degrees
	
	Output:
	    angle(s) in radians
	'''
	return np.pi/180. * theta


def rad2deg(theta):
	'''
	Convert from radians to degrees.
	
	Input:
	    theta    angle or numpy array of angles, in radians
	
	Output:
	    angle(s) in degrees
	'''
	return 180./np.pi * theta


def healmap_rasterize(m, nside, nest=True, lb_bounds=[0., 360., -90., 90.], size=[1000,1000], center_gal=False, return_theta_phi=False):
	'''
	Rasterize a healpix map in Cartesian projection.
	
	Input:
	    m           Healpix map, list of maps, or ndarray with first index
	                identifying separate maps.
	    nside       Healpix nside parameter.
	    nest        True if map is stored in nested healpix ordering.
	    lb_bounds   (l_min, l_max, b_min, b_max). Default: all l,b included.
	    size        (x_size, y_size). # of pixels in each dimension.
	    center_gal  If True, place rasterized pixel closest to l=0 at
	                center of image.
	    
	    return_theta_phi  If True, spherical theta and phi (physics
	                      convention) for each rasterized pixel are
	                      returned.
	
	Output:
	    img         Rasterized image of healpix map in Cartesian projection.
	    
	If return_theta_phi is True, the following are also returned:
	    theta       Spherical theta coordinate in rad (physics convention).
	    phi         Spherical phi coordinate in rad (physics convention).
	'''
	
	# Make grid of pixels to plot
	xsize, ysize = size
	l, b = np.mgrid[0:xsize, 0:ysize].astype(np.float32) + 0.5
	#print xsize, ysize
	#print l.shape, b.shape
	l = lb_bounds[1] - (lb_bounds[1] - lb_bounds[0]) * l / float(xsize)
	b = lb_bounds[2] + (lb_bounds[3] - lb_bounds[2]) * b / float(ysize)
	theta, phi = lb2thetaphi(l, b)
	del l, b
	#print size
	
	# Convert coordinates to healpix pixels and create 2D map
	pix = hp.ang2pix(nside, theta, phi, nest=nest)
	img = None
	if type(m) == list:
		img = []
		for submap in m:
			img.append(m[pix])
			img[-1].shape = (xsize, ysize)
	elif len(m.shape) == 2:
		img = m[:,pix]
		img.shape = (img.shape[0], xsize, ysize)
	else:
		img = m[pix]
		img.shape = (xsize, ysize)
	del pix
	
	# Center map on l=0
	if center_gal:
		phi[phi >= 360.] -= 360.
		shift = int(np.round(xsize/2. - np.unravel_index(np.argmin(np.abs(phi)), img.shape)[0]))
		#print 'Rolling by %d pixels (%.2f%% of xsize)' % (shift, 100.*float(shift)/xsize)
		if type(m) == list:
			for i,subimg in enumerate(img):
				img[i] = np.roll(subimg, shift, axis=0)
		elif len(m.shape) == 2:
			img = np.roll(img, shift, axis=1)
		else:
			img = np.roll(img, shift, axis=0)
	
	if return_theta_phi:
		return img, theta, phi
	else:
		return img


def healmap_to_axes(ax, m, nside, nest=True, lb_bounds=[0., 360., -90., 90.], size=[1000,1000], center_gal=False, **kwargs):
	'''
	Plot healpix map to the given pyplot axes. This function borrows
	liberally from Eddie Schlafly's util_efs.imshow.
	
	Input:
	    ax          Axes to which to plot healpix map.
	    m           Healpix map.
	    nside       Healpix nside parameter.
	    nest        True if map is stored in nested healpix ordering.
	    lb_bounds   (l_min, l_max, b_min, b_max). Default: all l,b included.
	    size        (x_size, y_size). # of pixels in each dimension.
	    center_gal  If True, place rasterized pixel closest to l=0 at
	                center of image.
	    **kwargs    Additional parameters to pass to pyplot.imshow.
	
	Output:
	    image       Image object returned by ax.imshow. This can be used,
	                for example, to create a colorbar.
	'''
	
	img, theta, phi = healmap_rasterize(m, nside, nest, lb_bounds, size, center_gal, True)
	
	lb_bounds_internal = np.array(lb_bounds)
	
	# Center map on l=0
	if center_gal:
		lb_bounds_internal[0] -= 180.
		lb_bounds_internal[1] -= 180.
	
	# Handle special case where the axes use the Mollweide projection
	if ax.name == 'mollweide':
		if not center_gal:
			lb_bounds_internal[0] -= 180.
			lb_bounds_internal[1] -= 180.
		if (np.abs(lb_bounds_internal[0] + 180.) > 0.001) or (np.abs(lb_bounds_internal[1] - 180.) > 0.001) or (np.abs(lb_bounds_internal[2] + 90.) > 0.001) or (np.abs(lb_bounds_internal[3] - 90.) > 0.001):
			print 'Warning: Mollweide projection requires lb_bounds = (0., 360., -90., 90.).'
		lb_bounds_internal = list(deg2rad(np.array(lb_bounds_internal)))
	
	tmp_l_max = lb_bounds_internal[1]
	lb_bounds_internal[1] = lb_bounds_internal[0]
	lb_bounds_internal[0] = tmp_l_max
	
	# Plot to given axes
	if (ax.name != 'mollweide') and ('interpolation' not in kwargs):
		kwargs['interpolation'] = 'nearest'
	if 'vmin' not in kwargs:
		kwargs['vmin'] = np.min(img[np.isfinite(img)])
	if 'vmax' not in kwargs:
		kwargs['vmax'] = np.max(img[np.isfinite(img)])
	if 'aspect' not in kwargs:
		kwargs['aspect'] = 'auto'
	if 'origin' in kwargs:
		print "Ignoring option 'origin'."
	if 'extent' in kwargs:
		print "Ignoring option 'extent'."
	kwargs['origin'] = 'lower'
	kwargs['extent'] = lb_bounds_internal
	image = ax.imshow(img.T, **kwargs)
	
	return image


class ExtinctionMap():
	'''
	Class representing a set of extinction maps at increasing distance.
	The maps are stored internally in healpix ordering.
	'''
	
	def __init__(self):
		'''
		Initiate a dummy extinction map with nside=1.
		'''
		self.nested = True
		self.nside = 1
		self.Ar = np.zeros((1, hp.nside2npix(self.nside)), dtype=np.float64)
		self.mu = np.zeros(1, dtype=np.float64)
		self.N_stars = np.zeros(hp.nside2npix(self.nside), dtype=np.uint32)
		self.measure = np.zeros(hp.nside2npix(self.nside), dtype=np.float64)
	
	def __init__(self, fname, FITS=True, nside=512, nested=True):
		if FITS:
			self.load(fname)
		else:
			self.load_from_binaries(fname, nside, nested)
	
	def save(self, fname):
		'''
		Save the extinction map to a FITS file.
		'''
		# Primary HDU with list of distance moduli
		primary = pyfits.PrimaryHDU(self.mu)
		primary.header.update('PIXTYPE', 'HEALPIX')
		if self.nested:
			primary.header.update('ORDERING', 'NESTED')
		else:
			primary.header.update('ORDERING', 'RING')
		primary.header.update('NSIDE', self.nside)
		primary.header.update('FIRSTPIX', 0)
		primary.header.update('LASTPIX', hp.nside2npix(self.nside))
		primary.header.update('UNITS', 'MAGS')
		primary.header.update('NDIST', self.Ar.shape[0])
		primary.header.update('DESC', 'DISTMODS')
		primary.header.update('DTYPE', 'FLOAT64')
		hdu = []
		hdu.append(primary)
		
		# Image HDUs with # of stars and fit measure
		img = pyfits.ImageHDU(self.N_stars)
		img.header.update('DESC', 'NSTARS')
		img.header.update('DTYPE', 'UINT32')
		hdu.append(img)
		img = pyfits.ImageHDU(self.measure)
		img.header.update('DESC', 'MEASURE')
		img.header.update('DTYPE', 'FLOAT64')
		hdu.append(img)
		
		# Image HDUs with Ar
		for mu,Ar in zip(self.mu, self.Ar):
			img = pyfits.ImageHDU(Ar)
			img.header.update('DESC', 'EXTINCTN')
			img.header.update('DTYPE', 'FLOAT64')
			img.header.update('BAND', 'PS r')
			img.header.update('UNITS', 'MAGS')
			img.header.update('DISTMOD', mu)
			hdu.append(img)
		
		hdulist = pyfits.HDUList(hdu)
		hdulist.writeto(abspath(fname), clobber=True)
	
	def load(self, fname):
		'''
		Load an extinction map from a FITS file.
		'''
		try:
			f = pyfits.open(abspath(fname))
		except:
			print 'Could not load %s' % fname
			return
		if f[0].header['ORDERING'] == 'NESTED':
			self.nested = True
		else:
			self.nested = False
		self.nside = int(f[0].header['NSIDE'])
		self.mu = f[0].data
		self.N_stars = f[1].data
		self.measure = f[2].data
		self.Ar = np.empty((int(f[0].header['NDIST']), hp.nside2npix(self.nside)), dtype=np.float64)
		for i,img in enumerate(f[3:]):
			self.Ar[i,:] = img.data
		f.close()
	
	def load_from_binaries(self, fname, nside=512, nested=True):
		'''
		Load an extinction map from a set of binary files produced by
		fit_pdfs.py.
		'''
		self.nside = nside
		self.nested = nested
		self.mu = None
		self.Ar = None
		self.N_stars = np.zeros(hp.nside2npix(self.nside), dtype=np.uint32)
		self.measure = np.empty(hp.nside2npix(self.nside), dtype=np.float64)
		self.measure.fill(np.NaN)
		
		if type(fname) is str:
			fname = [fname]
		
		# Store (DM, Ar) fit for each healpix pixel
		for filename in fname:
			#print 'Opening %s ...' % filename
			f = open(abspath(filename), 'rb')
			while True:
				try:	# Load in individual pixels until the EOF, or until something goes wrong
					pix_index = np.fromfile(f, dtype=np.uint64, count=1)[0]
					self.N_stars[pix_index] = np.fromfile(f, dtype=np.uint32, count=1)[0]
					self.measure[pix_index] = np.fromfile(f, dtype=np.float64, count=1)[0]
					success = np.fromfile(f, dtype=np.uint16, count=1)[0]
					N_regions = np.fromfile(f, dtype=np.uint16, count=1)[0]
					line_int = np.fromfile(f, dtype=np.float64, count=self.N_stars[pix_index])
					if self.mu == None:
						self.mu = np.fromfile(f, dtype=np.float64, count=N_regions+1)
						self.Ar = np.empty((N_regions+1, hp.nside2npix(self.nside)), dtype=np.float64)
						self.Ar.fill(np.NaN)
					else:
						mu_anchors = np.fromfile(f, dtype=np.float64, count=N_regions+1)
					self.Ar[:, pix_index] = np.fromfile(f, dtype=np.float64, count=N_regions+1)
				except:
					f.close()
					break
			f.close()
	
	def evaluate(self, mu_eval, pix_index=None):
		'''
		Evaluate Ar at the given distance modulus, or list of distance
		moduli, mu_eval.
		'''
		if type(mu_eval) not in [list, np.ndarray]:
			mu_eval = [mu_eval]
		
		query_map = self.Ar
		if pix_index != None:
			query_map = self.Ar[:,pix_index]
		
		# Create an empty map for each value of mu
		Ar_map = None
		if (pix_index == None) or (type(pix_index) in [list, np.ndarray]):
			Ar_map = np.empty((len(mu_eval), query_map.shape[1]), dtype=np.float64)
		else:
			Ar_map = np.empty(len(mu_eval), dtype=np.float64)
		Ar_map.fill(np.NaN)
		
		for k,m in enumerate(mu_eval):
			if (m >= self.mu[0]) and (m <= self.mu[-1]):
				for i,mu_anchor in enumerate(self.mu[1:]):
					if mu_anchor >= m:
						slope = (query_map[i+1] - query_map[i]) / (self.mu[i+1] - self.mu[i])
						Ar_map[k] = query_map[i] + slope * (m - self.mu[i])
						break
		
		return Ar_map
	
	def chi2dof(self):
		'''
		Return something like the chi^2/d.o.f. This is not strictly the
		same as chi^2/d.o.f., and is not normalized, but should give a
		measure of the relative goodness of fit in different pixels.
		'''
		return np.divide(self.measure, self.N_stars - len(self.mu) - 1)
	
	def npix(self):
		'''
		Return # of valid pixels in map (i.e. where Ar is not stored
		as the default, NaN).
		'''
		return np.sum(np.isfinite(self.Ar[0]))
	
	def valid_pixels(self):
		'''
		Return the healpix numbers of the pixels which have been loaded.
		'''
		return np.where(np.isfinite(self.Ar[0]))[0]
	
	def pixel_bounds(self):
		'''
		Determine the bounds in Galactic longitude and latitude of the
		valid pixels in the map.
		
		Output:
		    l_min, l_max, b_min, b_max
		'''
		pix_index = self.valid_pixels()
		theta, phi = hp.pix2ang(self.nside, pix_index, nest=self.nested)
		l, b = thetaphi2lb(theta, phi)
		del theta, phi
		return np.min(l), np.max(l), np.min(b), np.max(b)
	
	def Ar_percentile(self, percentile, lb_bounds=None, mu_eval=None):
		'''
		Determine the threshold in Ar below which a given percentage of
		pixels lie.
		
		Inputs:
		    percentile  Percentile (0-100), or list of percentiles
		    lb_bounds   l_min, l_max, b_min, b_max
		    mu_eval     Distance modulus at which to determine percentile
		
		Output:
		    Ar threshold, or list of thresholds
		'''
		pix_index = self.valid_pixels()
		theta, phi = hp.pix2ang(self.nside, pix_index, nest=self.nested)
		l, b = thetaphi2lb(theta, phi)
		del theta, phi
		Ar_map_clipped = None
		if mu_eval == None:
			Ar_map_clipped = self.Ar[-1]
		else:
			Ar_map_clipped = self.evaluate(mu_eval)[0]
		if lb_bounds != None:
			pix_mask = (l >= lb_bounds[0]) & (l <= lb_bounds[1]) & (b >= lb_bounds[2]) & (b <= lb_bounds[3])
			Ar_map_clipped = Ar_map_clipped[pix_index[pix_mask]]
		else:
			Ar_map_clipped = Ar_map_clipped[pix_index]
		Ar_map_clipped = Ar_map_clipped[np.isfinite(Ar_map_clipped)]
		Ar_map_clipped.sort()
		if type(percentile) is list:
			percentile = np.array(percentile)
		index = percentile/100. * (Ar_map_clipped.size-1.)
		if type(index) is np.ndarray:
			index = index.astype(np.uint32)
		else:
			index = int(index)
		return Ar_map_clipped[index]
		
	def rasterize(self, mu_eval, size='native', lb_bounds='auto', center_gal=False, **kwargs):
		'''
		Rasterize the extinction map.
		
		Input:
			mu_eval     Distance modulus or moduli at which to rasterize map
			size        Width and height of rasterized image(s) to produce.
			            If 'native' or 'auto' is given, then estimate
			            correct resolution given nside parameter of map
			            and the (l,b) bounds of the image.
			lb_bounds   Bounds in Galactic l and b (in degrees) to rasterize.
			            If 'auto' is given, then the bounds are set to
			            include all the valid pixels. If 'full' is given,
			            then the bounds are set to (0, 360, -90, 90).
			center_gal  If True, place rasterized pixel closest to l=0 at
	                    center of image.
			
			**kwargs    Additional parameters to pass to healmap_rasterize.
		
		Output:
			img         Rasterized image of healpix map in Cartesian projection.
		'''
		# Interpret size and lb_bounds
		resolution = rad2deg(hp.pixelfunc.nside2resol(self.nside))
		if type(lb_bounds) == str:
			if lb_bounds.lower() == 'auto':
				lb_bounds = list(self.pixel_bounds())
				lb_bounds[0] -= resolution
				lb_bounds[1] += resolution
				lb_bounds[2] -= resolution
				lb_bounds[3] += resolution
				if lb_bounds[2] < -90.:
					lb_bounds[2] = -90.
				if lb_bounds[3] > 90.:
					lb_bounds[3] = 90.
			elif lb_bounds.lower() == 'full':
				lb_bounds = [0., 360., -90., 90.]
			else:
				raise ValueError('Unrecognized bounds option: %s' % lb_bounds)
		if type(size) == str:
			if size.lower() in ['native', 'auto']:
				size = [int(3.*(lb_bounds[1] - lb_bounds[0])/resolution), int(3.*(lb_bounds[3] - lb_bounds[2])/resolution)]
			else:
				raise ValueError('Unrecognized size option: %s' % size)
		if 'nside' in kwargs:
			print "Ignoring option 'nside'."
		if 'nest' in kwargs:
			print "Ignoring option 'nest'."
		
		if center_gal:
			lb_bounds[0] -= 180.
			lb_bounds[1] -= 180.
		
		# Evaluate healpix map and pass to rasterizing function
		m = self.evaluate(mu_eval)
		return healmap_rasterize(m, self.nside, nest=self.nested, lb_bounds=lb_bounds, size=size, center_gal=center_gal, **kwargs), lb_bounds
	
	def to_axes(self, ax, mu_eval, size='native', lb_bounds='auto', center_gal=False, log_scale=False, diff=False, **kwargs):
		'''
		Plot the extinction map to the given axes.
		
		Input:
		    ax          Axes or list of axes two which to plot map.
			mu_eval     Distance modulus or moduli at which to rasterize map.
			size        Width and height of rasterized image(s) to produce.
			            If 'native' or 'auto' is given, then estimate
			            correct resolution given nside parameter of map
			            and the (l,b) bounds of the image. (Default: 'native')
			lb_bounds   Bounds in Galactic l and b (in degrees) to rasterize.
			            If 'auto' is given, then the bounds are set to
			            include all the valid pixels. If 'full' is given,
			            then the bounds are set to (0, 360, -90, 90).
			            (Default: 'auto')
			center_gal  If True, place rasterized pixel closest to l=0 at
	                    center of image. (Default: False)
	        log_scale   Use a logarithmic scale. (Default: False)
	        diff        Plot the difference between extinction at subsequent
	                    distances. (Default: False)
			
			**kwargs    Additional parameters to pass to pyplot.imshow.
		
		Output:
			image       Image object returned by ax.imshow. This can be used,
						for example, to create a colorbar.
		'''
		# Rasterize the map
		img, lb_bounds = self.rasterize(mu_eval, size=size, lb_bounds=lb_bounds, center_gal=center_gal)
		
		if type(mu_eval) not in [list, np.ndarray]:
			mu_eval = [mu_eval]
		if len(img.shape) == 2:
			img.shape = (1, img.shape[0], img.shape[1])
		elif diff:
			img[1:] = img[1:] - img[:-1]
		if type(ax) != list:
			ax = [ax]
		
		# Set imshow options
		vmin, vmax = None, None
		if 'vmin' not in kwargs:
			vmin, vmax = self.Ar_percentile([5, 95], lb_bounds=lb_bounds, mu_eval=np.max(mu_eval))
			if log_scale:
				kwargs['vmin'] = np.log(vmin) #np.log(np.min(img[np.isfinite(img)]))
			else:
				kwargs['vmin'] = vmin #np.min(img[np.isfinite(img)])
		if 'vmax' not in kwargs:
			if log_scale:
				kwargs['vmax'] = np.log(vmax) #np.log(np.max(img[np.isfinite(img)]))
			else:
				kwargs['vmax'] = vmax #np.max(img[np.isfinite(img)])
		if 'aspect' not in kwargs:
			kwargs['aspect'] = 'auto'
		if 'origin' in kwargs:
			print "Ignoring option 'origin'."
		if 'extent' in kwargs:
			print "Ignoring option 'extent'."
		kwargs['origin'] = 'lower'
		
		if center_gal:
			lb_bounds[0] -= 180.
			lb_bounds[1] -= 180.
		
		# Plot each distance to a different axis
		image = []
		for subax, subimg in zip(ax, img):
			lb_bounds_internal = np.array(lb_bounds)
			
			# Handle special case where the axes use the Mollweide projection
			if subax.name == 'mollweide':
				if not center_gal:
					lb_bounds_internal[0] -= 180.
					lb_bounds_internal[1] -= 180.
				if (np.abs(lb_bounds_internal[0] + 180.) > 0.001) or (np.abs(lb_bounds_internal[1] - 180.) > 0.001) or (np.abs(lb_bounds_internal[2] + 90.) > 0.001) or (np.abs(lb_bounds_internal[3] - 90.) > 0.001):
					print 'Warning: Mollweide projection requires lb_bounds = (0., 360., -90., 90.).'
				lb_bounds_internal = list(deg2rad(np.array(lb_bounds_internal)))
			
			tmp_l_max = lb_bounds_internal[1]
			lb_bounds_internal[1] = lb_bounds_internal[0]
			lb_bounds_internal[0] = tmp_l_max
			
			# Plot to given axes
			if (subax.name != 'mollweide') and ('interpolation' not in kwargs):
				kwargs['interpolation'] = 'nearest'
			kwargs['extent'] = lb_bounds_internal
			if log_scale:
				image.append(subax.imshow(np.log(subimg.T), **kwargs))
			else:
				image.append(subax.imshow(subimg.T, **kwargs))
		
		return image



def main():
	fname = ['../output/gal_plane_reddening_pt1_500%d.dat' % i for i in range(3)]
	m = ExtinctionMap(fname, FITS=False)
	
	# Set up figure
	mplib.rc('text', usetex=True)
	mplib.rc('axes', grid=False)
	fig = plt.figure(figsize=(7,5))
	ax = []
	for i in range(6):
		ax.append(fig.add_subplot(2,3,i+1))#, projection='mollweide'))
		y, x = np.unravel_index(i, (2,3))
		if y != 1:
			ax[-1].set_xticklabels([])
		if x != 0:
			ax[-1].set_yticklabels([])
	
	# Plot map to axes
	image = m.to_axes(ax, np.linspace(5., 15., 6), log_scale=False, diff=False, vmin=0.)
	
	# Add color bar
	fig.subplots_adjust(wspace=0., hspace=0., right=0.88)
	cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
	cb = fig.colorbar(image, cax=cax)
	
	fits_fname = '../output/test.fits'
	#m.save(fits_fname)
	
	import fit_pdfs as fp
	pix_index = np.nanargmax(m.evaluate(15.))
	mu_anchors = np.linspace(5., 15., 6)
	Delta_Ar, weight = fp.get_neighbors(fits_fname, pix_index, mu_anchors=mu_anchors)
	#print Delta_Ar
	print weight
	for mu, DAr in zip(mu_anchors, Delta_Ar):
		print mu, DAr
	
	plt.show()
	
	return 0

if __name__ == '__main__':
	main()

