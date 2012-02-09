#!/usr/bin/env python2.7
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


from math import pi, sqrt, exp, log
from numpy import linspace, empty
from natspline import clsNaturalSpline

LF_file = '/home/greg/projects/galstar/data/MrLF.MSandRGB_v1.0.dat'

class clsLOSModel:
	P_DM = None		# Spline with (P(DM), DM)
	P_Mr = None		# Spline with (P(Mr), Mr)
	cos_b = None
	sin_b = None
	cos_l = None
	sin_l = None
	r_0 = None
	z_0 = None
	
	# TODO: Initialize the following variables
	h_1 = None
	l_1 = None
	
	def __init__(self, l, b, lf_fn):
		self.r_0 = 8000.
		self.z_0 = 25.
		self.h_1 = 2150.
		self. l_1 = 245.
		
		l_rad = l*pi/180.
		b_rad = b*pi/180.
		self.cos_b = cos(b_rad)
		self.sin_b = sin(b_rad)
		self.cos_l = cos(l_rad)
		self.sin_l = sin(l_rad)
		
		self.load_lf(lf_fn)
		self.init_DM()
	
	def load_LF(self, luminosity_fn):
		# Read in the luminosity function from file
		f = open(luminosity_fn)
		p_Mr_list = []
		Mr_list = []
		for line in f:
			tmp = line.lstrip().rstrip()
			if tmp[0] != "#":
				tmp = tmp.split()
				Mr_list.append(tmp[0])
				p_Mr_list.append(tmp[1])
		f.close()
		# Calculate and spline the cumulative distribution P(Mr)
		P_Mr_list = empty(len(p_Mr_list))
		P_Mr_list[0] = 0.
		for i,p in enumerate(p_Mr_list):
			P_Mr_list[i+1] = P_Mr_list[i] + p
		norm = P_Mr_list[-1]
		for i in range(len(P_Mr_list)):
			P_Mr_list[i] /= norm
		self.P_Mr = clsNaturalSpline(P_Mr_list, Mr_list)
	
	def CartesianPos(self, DM):
		d = 10.**(DM/5.+1.)
		x = R_0 - cos_l*cos_b*d
		y = -sin_l*cos_b*d
		z = self.z_0 + sin_b*d
		return x,y,z
	
	def dn(self, DM):
		x,y,z = CartesianPos(self, DM)
		r = sqrt(x*x+y*y)
		return exp(-(abs(z+self.z_0) - abs(self.z_0))/self.h_1 - (r-self.r_0)/self.l_1)
	
	def init_DM(self):
		return


def main():
	
	return 0

if __name__ == '__main__':
	main()

