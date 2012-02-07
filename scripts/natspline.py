#!/usr/bin/env python
#
#       natspline.py
#       
#       Copyright 2010 Gregory Green <greg@greg-laptop>
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

from numpy import matrix

# Natural cubic spline to a set of knots defined by two x_list and y_list
class clsNaturalSpline:
	# Initialize the spline using the lists x_list and y_list
	def __init__(self, x_list, y_list):
		self.set_spline(x_list, y_list)
	# Reset the spline using the lists x_list and y_list
	def set_spline(self, x_list, y_list):
		self.x_list = x_list
		
		M = [0.]
		tmp = gen_M_matrix(x_list, y_list).T.tolist()[0]
		for m in tmp:
			M.append(m)
		M.append(0.)
		
		self.a_list = []
		self.b_list = []
		self.c_list = []
		self.d_list = []
		
		for i in range(len(y_list)-1):
			h = x_list[i+1] - x_list[i]
			self.a_list.append((M[i+1]-M[i])/6/h)
			self.b_list.append(M[i]/2)
			self.c_list.append((y_list[i+1]-y_list[i])/h - (M[i+1]+2*M[i])*h/6)
			self.d_list.append(y_list[i])
	# Returns f(x) if x belongs to [x_0, x_n]
	#	Else, returns None
	def get_y(self, x):
		for n in range(len(self.x_list)):
			if (x >= self.x_list[n]) and (x <= self.x_list[n+1]):
				return self.a_list[n]*(x-self.x_list[n])**3. + self.b_list[n]*(x-self.x_list[n])**2. + self.c_list[n]*(x-self.x_list[n]) + self.d_list[n]
		return None
	# Returns f'(x) if x belongs to [x_0, x_n]
	#	Else, returns None
	def get_dy(self, x):
		for n in range(len(self.x_list)):
			if (x >= self.x_list[n]) and (x <= self.x_list[n+1]):
				return 3.*self.a_list[n]*(x-self.x_list[n])**2. + 2.*self.b_list[n]*(x-self.x_list[n]) + self.c_list[n]
		return None
	# Returns f''(x) if x belongs to [x_0, x_n]
	#	Else, returns None
	def get_d2y(self, x):
		for n in range(len(self.x_list)):
			if (x >= self.x_list[n]) and (x <= self.x_list[n+1]):
				return 6.*self.a_list[n]*(x-self.x_list[n]) + 2.*self.b_list[n]
		return None

def gen_A_matrix(size):
	A = matrix([[0 for i in range(size)] for n in range(size)])
	for i in range(1,size):
		A[i,i] = 4.
		A[i-1,i] = 1.
		A[i,i-1] = 1.
	A[0,0] = 4.
	return A

def gen_f_matrix(x_list, y_list):
	f_list = []
	for i in range(len(y_list)-2):
		f_list.append([6./(x_list[i+1]-x_list[i])**2.*(y_list[i]-2.*y_list[i+1]+y_list[i+2])])
	return matrix(f_list)

def gen_M_matrix(x_list, y_list):
	A = gen_A_matrix(len(y_list)-2)
	f = gen_f_matrix(x_list, y_list)
	return A.I * f
