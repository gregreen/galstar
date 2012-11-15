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
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.integrate import quad

import matplotlib.pyplot as plt
import matplotlib as mplib

from model import TGalacticModel


class TSample1D:
	'''
	Draw samples from a 1D probability density function.
	'''
	def __init__(self, f, x_min, x_max, N=100, M=1000):
		x = np.linspace(x_min, x_max, N)
		try:
			p_x = f(x)
		except:
			p_x = [f(xx) for xx in x]
		P = np.zeros(N, dtype='f4')
		for i in xrange(1,N-1):
			P[i] = P[i-1] + 0.5*p_x[i-1]+0.5*p_x[i]
		P[-1] = P[-2] + 0.5*p_x[-1]
		P /= np.sum(p_x)
		P[-1] = 1.
		if N < M:
			P_spl = InterpolatedUnivariateSpline(x, P)
			x = np.linspace(x_min, x_max, M)
			P = P_spl(x)
			P[0] = 0.
			P[-1] = 1.
		self.x = interp1d(P, x, kind='linear')
	
	def __call__(self, N=1):
		P = np.random.random(N)
		return self.x(P)
	
	def get_x(self, P):
		return self.x(P)


def draw_from_model(l, b, N, Ar=0.5, r_max=23., Ar_of_mu=None):
	dtype = [('DM', 'f8'), ('Ar', 'f8'), ('Mr', 'f8'), ('FeH', 'f8')]
	ret = np.empty(N, dtype=dtype)
	
	l = np.pi/180. * l
	b = np.pi/180. * b
	cos_l, sin_l = np.cos(l), np.sin(l)
	cos_b, sin_b = np.cos(b), np.sin(b)
	
	model = TGalacticModel()
	
	mu_max = r_max - model.Mr_min + 0.5
	mu_min = min(1., mu_max-15.)
	Mr_max = min(r_max, model.Mr_max)
	dN_dDM = lambda mu: model.dn_dDM(mu, cos_l, sin_l, cos_b, sin_b)
	draw_mu = TSample1D(dN_dDM, mu_min, mu_max, 500, 10000)
	draw_Mr = TSample1D(model.LF, model.Mr_min, Mr_max, 10000, 1)
	
	idx = np.ones(N, dtype=np.bool)
	while np.any(idx):
		size = np.sum(idx)
		ret['DM'][idx] = draw_mu(size)
		ret['Mr'][idx] = draw_Mr(size)
		ret['Ar'][idx] = 0.
		if Ar_of_mu != None:
			ret['Ar'][idx] += Ar_of_mu(ret['DM'][idx]) + np.random.normal(scale=Ar, size=size)
		ret['Ar'][idx] += Ar * np.random.chisquare(1., size)
		idx = (ret['Mr'] + ret['DM'] + ret['Ar'] > r_max) | (ret['Ar'] < 0.)
		#print np.sum(idx), N
	
	x, y, z = model.Cartesian_coords(ret['DM'], cos_l, sin_l,
	                                                       cos_b, sin_b)
	
	halo = np.random.random(N) < model.f_halo(ret['DM'], cos_l, sin_l,
	                                                       cos_b, sin_b)
	thin = ~halo & (np.random.random(N) < 0.63)
	thick = ~halo & ~thin
	
	#FeH = np.empty(N, dtype='f8')
	
	idx = halo
	while np.any(idx):
		ret['FeH'][idx] = np.random.normal(-1.46, 0.3, size=np.sum(idx))
		idx = (ret['FeH'] <= -2.5) | (ret['FeH'] >= 0.)
	
	idx = thin
	while np.any(idx):
		ret['FeH'][idx] = np.random.normal(model.mu_FeH_D(z[idx])-0.067,
		                                          0.2, size=np.sum(idx))
		idx = (ret['FeH'] <= -2.5) | (ret['FeH'] >= 0.)
	
	idx = thick
	while np.any(idx):
		ret['FeH'][idx] = np.random.normal(model.mu_FeH_D(z[idx])-0.067+0.14,
		                                          0.2, size=np.sum(idx))
		idx = (ret['FeH'] <= -2.5) | (ret['FeH'] >= 0.)
	
	return ret


def draw_flat(N, Ar=0.5):
	dtype = [('DM', 'f8'), ('Ar', 'f8'), ('Mr', 'f8'), ('FeH', 'f8')]
	ret = np.empty(N, dtype=dtype)
	
	idx = np.ones(N, dtype=np.bool)
	while np.any(idx):
		ret['DM'][idx] = np.random.rand(N) * 13.5 + 5.5
		ret['Ar'][idx] = np.random.rand(N) * 2. * Ar
		ret['Mr'][idx] = np.random.rand(N) * 20. - 0.8
		idx = (ret['DM'] + ret['Ar'] + ret['Mr'] > 23.)
	
	ret['FeH'] = np.random.rand(N) * 2.4 - 2.45
	
	return ret


def main():
	parser = argparse.ArgumentParser(prog='gen_test_input.py',
	                                 description='Generates test input file for galstar.',
	                                 add_help=True)
	parser.add_argument('N', type=int, help='# of stars to generate.')
	parser.add_argument('-lb', '--gal-lb', type=float, nargs=2,
	                    metavar='deg', default=(90., 10.),
	                    help='Galactic latitude and longitude, in degrees.')
	parser.add_argument('-Ar', '--mean-Ar', type=float, default=0.5,
	                    metavar='mags', help='Mean r-band extinction.')
	parser.add_argument('-cl', '--clouds', type=float, nargs='+',
	                    default=None, metavar='mu Delta_Ar',
	                    help='Place clouds of reddening Delta_Ar at distances mu')
	parser.add_argument('-r', '--max-r', type=float, default=23.,
	                    metavar='mags', help='Limiting apparent r-band magnitude.')
	parser.add_argument('-flat', '--flat', action='store_true',
	                    help='Draw parameters from flat distribution')
	parser.add_argument('-sh', '--show', action='store_true',
	                    help='Plot distribution of DM, Mr and Ar.')
	#parser.add_argument('-b', '--binary', action='store_true', help='Generate binary stars.')
	if 'python' in sys.argv[0]:
		offset = 2
	else:
		offset = 1
	values = parser.parse_args(sys.argv[offset:])
	
	Ar_of_mu = None
	if values.clouds != None:
		mu = np.linspace(-5., 35., 1000)
		dmu = mu[1] - mu[0]
		Delta_Ar = 0.01 * dmu * np.ones(mu.size)
		for i in range(len(values.clouds)/2):
			s = 0.05
			m = values.clouds[2*i]
			A = values.clouds[2*i+1]
			Delta_Ar += + A/np.sqrt(2.*np.pi)/s*np.exp(-(mu-m)*(mu-m)/2./s/s)
		Ar = np.cumsum(Delta_Ar) * dmu
		Ar_of_mu = InterpolatedUnivariateSpline(mu, Ar)
		mu = np.linspace(5., 20., 1000)
		Ar = Ar_of_mu(mu)
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.plot(mu, Ar)
		plt.show()
		#print Ar
	
	params = None
	if values.flat:
		params = draw_flat(values.N, Ar=values.mean_Ar,
		                   r_max=values.max_r, Ar_of_mu=Ar_of_mu)
	else:
		params = draw_from_model(values.gal_lb[0], values.gal_lb[1],
		                         values.N, Ar=values.mean_Ar,
		                         r_max=values.max_r, Ar_of_mu=Ar_of_mu)
	
	header = '''# Format:
# l  b
# DM  Ar  Mr  FeH
# DM  Ar  Mr  FeH
# DM  Ar  Mr  FeH
# ...'''
	print header
	print '%.3f  %.3f' % (values.gal_lb[0], values.gal_lb[1])
	for p in params:
		print '%.3f  %.3f  %.3f  %.3f' % (p['DM'], p['Ar'], p['Mr'], p['FeH'])
	
	if values.show:
		model = TGalacticModel()
		l = np.pi/180. * values.gal_lb[0]
		b = np.pi/180. * values.gal_lb[1]
		cos_l, sin_l = np.cos(l), np.sin(l)
		cos_b, sin_b = np.cos(b), np.sin(b)
		dN_dDM = lambda mu: model.dn_dDM(mu, cos_l, sin_l, cos_b, sin_b)
		
		mplib.rc('text', usetex=True)
		
		fig = plt.figure(figsize=(6,4), dpi=300)
		
		ax = fig.add_subplot(2,2,1)
		ax.hist(params['DM'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(xlim[0], xlim[1], 1000)
		ax.plot(x, dN_dDM(x)/quad(dN_dDM, 1., 25.)[0], 'g-', lw=1.3, alpha=0.5)
		ax.set_xlim(xlim)
		ax.set_xlabel(r'$\mu$', fontsize=14)
		
		ax = fig.add_subplot(2,2,2)
		ax.hist(params['Mr'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(model.Mr_min, model.Mr_max, 1000)
		ax.plot(x, model.LF(x)/quad(model.LF, x[0], x[-1], full_output=1)[0],
		                                               'g-', lw=1.3, alpha=0.5)
		ax.set_xlim(xlim)
		ax.set_xlabel(r'$M_{r}$', fontsize=14)
		
		ax = fig.add_subplot(2,2,3)
		ax.hist(params['Ar'], bins=100, normed=True, alpha=0.3)
		ax.set_xlabel(r'$A_{r}$', fontsize=14)
		
		ax = fig.add_subplot(2,2,4)
		ax.hist(params['FeH'], bins=100, normed=True, alpha=0.3)
		xlim = ax.get_xlim()
		x = np.linspace(xlim[0], xlim[1], 100)
		y = model.p_FeH_los(x, cos_l, sin_l, cos_b, sin_b)
		ax.plot(x, y, 'g-', lw=1.3, alpha=0.5)
		ax.set_xlabel(r'$\left[ Fe / H \right]$', fontsize=14)
		ax.set_xlim(xlim)
		
		fig.subplots_adjust(hspace=0.40, wspace=0.25,
		                    bottom=0.13, top=0.95,
		                    left=0.1, right=0.9)
		
		plt.show()
	
	return 0

if __name__ == '__main__':
	main()

