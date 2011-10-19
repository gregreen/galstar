
#ifndef _NDARRAY_H__
#define _NDARRAY_H__

#include <iostream>
#include <stdlib.h>
#include <assert.h>

/** ******************************************************************************************************
 *
 *	NDArray<T>
 *
 *	This is a general-purpose multidimensional array, which can hold elements of any class T. The
 *	array must be initialized with a list of unsigned integers giving the width of each dimension.
 *
 *	Ex.: Declaring the array.
 *		unsigned int width[3] = {10, 3, 12};
 *		NDArray<double> arr(&(width[0]), 3);
 *
 *	Elements are accessed in one of two ways. The first method is identical to the method used to
 *	access elements of multidimensional c-style arrays.
 *
 *	Ex.: Accessing elements using consecutive brackets.
 *		arr[8][1][3] = 3.5;
 *
 *	The second method of accessing elements is by calling get_element with an array of unsigned integers.
 *
 *	Ex.: Accessing elements using an array of unsigned integers.
 *		unsigned int index[3] = {8, 1, 3};
 *		arr.get_element(&(index[0]), 3) = 3.5;
 *
 *	This code was inspired by Giovanni Bavestrelli's article, "A Class Template for N-Dimensional
 *	Generic Resizable Arrays", published December 01, 2000 at <http://drdobbs.com/cpp/184401319>.
 * 
 *********************************************************************************************************/

template<class T> class NDArray;
template<class T> class sub_NDArray;

template<class T>
std::ostream& operator<<(std::ostream& out, const sub_NDArray<T> &rhs);


/** **************************************************************************************************************************************
// sub_NDArray<T> class prototype
/****************************************************************************************************************************************/

// sub_NDArray is used when accessing an element of NDArray using the method arr[#][#]...[#]
template<class T>
class sub_NDArray {
	T *const element;
	const unsigned int * width;
	const unsigned int * multiplier;
	unsigned int N;
	
	// Constructor and Destructor
	sub_NDArray<T>(T *_element, const unsigned int *const _width, const unsigned int *const _multiplier, unsigned int _N);
	
public:
	sub_NDArray<T> operator [](unsigned int index);
	bool operator ==(const NDArray<T> &rhs);			// Can be compared to an N-dimensional array
	bool operator ==(const sub_NDArray<T> &rhs);			// Can be compared to an N-dimensional slice of another array of higher dimension
	sub_NDArray<T>& operator =(const T &rhs);			// Only for zero-dimensional arrays (scalars)
	sub_NDArray<T>& operator =(const NDArray<T> &rhs);		// Can be copied from an array of dimension N
	sub_NDArray<T>& operator =(const sub_NDArray<T> &rhs);		// Can be copied from another N-dimensional slice of some array of higher dimension
	bool operator !=(const NDArray<T> &rhs) { return !(operator==(rhs)); }
	bool operator !=(const sub_NDArray<T> &rhs) { return !(operator==(rhs)); }
	
	T* operator &() const { assert(N == 0); return element; }
	operator T () const { assert(N == 0); return element[0]; }
	//operator T* () { assert(N == 1); return element; }
	
	template<class U>
	friend std::ostream& operator <<(std::ostream& out, const sub_NDArray<U> &rhs); // Standard output for zero-dimensional arrays
	
	friend class NDArray<T>;
};


/** **************************************************************************************************************************************
// sub_NDArray<T> class members
/****************************************************************************************************************************************/

template<class T>
sub_NDArray<T>::sub_NDArray(T *_element, const unsigned int *const _width, const unsigned int *const _multiplier, unsigned int _N)
	: element(_element) , width(_width) , multiplier(_multiplier), N(_N)
{ }

template<class T>
sub_NDArray<T> sub_NDArray<T>::operator [](unsigned int index) {
	assert(N != 0);
	assert(index < width[0]);
	sub_NDArray<T> sub_arr(&element[index*multiplier[0]], &width[1], &multiplier[1], N-1);
	return sub_arr;
}

template<class T>
bool sub_NDArray<T>::operator ==(const sub_NDArray<T>& rhs) {
	if(this == &rhs) { return true; }
	if(rhs.N != N) { return false; }
	unsigned int size = 1;
	for(unsigned int i=0; i<N; i++) {
		if(width[i] != rhs.width[i]) { return false; }
		size *= width[i];
	}
	for(unsigned int i=0; i<size; i++) {
		if(element[i] != rhs.element[i]) { return false; }
	}
	return true;
}

template<class T>
bool sub_NDArray<T>::operator ==(const NDArray<T>& rhs) {
	if(rhs.N != N) { return false; }
	for(unsigned int i=0; i<N; i++) {
		if(width[i] != rhs.width[i]) { return false; }
	}
	for(unsigned int i=0; i<rhs.size; i++) {
		if(element[i] != rhs.element[i]) { return false; }
	}
	return true;
}

template<class T>
sub_NDArray<T>& sub_NDArray<T>::operator =(const sub_NDArray<T> &rhs) {
	//if(this == &rhs) { return *this; }
	unsigned int size = 1;
	for(unsigned int i=0; i<N; i++) {
		assert(multiplier[i] == rhs.multiplier[i]);
		assert(width[i] == rhs.width[i]);
		size *= width[i];
	}
	for(unsigned int i=0; i<size; i++) { element[i] = rhs.element[i]; }
	return *this;
}

template<class T>
sub_NDArray<T>& sub_NDArray<T>::operator =(const NDArray<T> &rhs) {
	for(unsigned int i=0; i<N; i++) {
		assert(multiplier[i] == rhs.multiplier[i]);
		assert(width[i] == rhs.width[i]);
	}
	for(unsigned int i=0; i<rhs.size; i++) { element[i] = rhs.element[i]; }
	return *this;
}

template<class T>
sub_NDArray<T>& sub_NDArray<T>::operator =(const T &rhs) {
	assert(N == 0);
	element[0] = rhs;
	return *this;
}

template<class T>
std::ostream& operator<<(std::ostream& out, const sub_NDArray<T> &rhs)
{
	assert(rhs.N == 0);
	out << rhs.element[0];
	return out;
}


/** **************************************************************************************************************************************
// NDArray<T> class prototype
/****************************************************************************************************************************************/

template<class T>
class NDArray {
protected:
	// Protected data
	T *element;
	unsigned int size;		// # of elements
	unsigned int N;			// # of dimensions
	unsigned int *width;		// # of elements along each axis
	unsigned int *multiplier;	// used in calculating index of element in array
	
public:
	// Iterator class
	class iterator;
	
	// Constructor and Destructor
	NDArray<T>(unsigned int _N);
	NDArray<T>(unsigned int *_width, unsigned int _N);
	NDArray<T>(const NDArray<T> &original);
	NDArray<T>(const NDArray<T> &original, const unsigned int *const flatten_axes, unsigned int N_axes);
	NDArray<T>(const sub_NDArray<T> &original);
	~NDArray<T>();
	
	// Operators
	sub_NDArray<T> operator [](unsigned int index);				// Access element using [#][#]...[#]
	NDArray<T>&  operator =(const NDArray<T> &rhs);
	NDArray<T>&  operator =(const sub_NDArray<T> &rhs);		// Can be copied from an N-dimensional slice of another array of higher dimension
	bool operator ==(const NDArray<T> &rhs) const;
	bool operator ==(const sub_NDArray<T> &rhs) const;		// Can be compared to an N-dimensional slice of another array of higher dimension
	bool operator !=(const NDArray<T> &rhs) const { return !(operator==(rhs)); }
	bool operator !=(const sub_NDArray<T> &rhs) const { return !(operator==(rhs)); }
	
	// Mutators
	void resize(const unsigned int *const _width, unsigned int _N);					// Resize the array, losing its contents
	void fill(const T &value) { for(unsigned int i=0; i<size; i++) { element[i] = value; } }	// Fill the array with a constant value
	
	template<class TParams> void fill(T (*func)(const unsigned int *const pos, unsigned int _N, const TParams &params), const TParams &params);
	void fill(T (*func)(const unsigned int *const pos, unsigned int _N));
	
	/*
	template<class TParams> void fill(T (*func)(const unsigned int *const pos, const TParams &params), const TParams &params) {	// Fill the array with a function of position
		unsigned int pos[N];
		for(iterator it=begin(); it!=end(); ++it) {
			it.get_pos(&pos[0]);
			*it = func(&pos[0], params);
		}
	}
	void fill(T (*func)(const unsigned int *const pos)) {
		iterator it = begin();
		for(; it != end(); ++it) { *it = func(it.get_pos()); }
	}
	*/
	
	// Accessors
	const unsigned int &get_width(unsigned int index) const { return width[index]; }
	const unsigned int &get_size() const { return size; }
	const unsigned int &get_N_dim() const { return N; }
	T& get_element(unsigned int index) const { return element[index]; }
	T& get_element(const unsigned int *const index, unsigned int _N) const;	// Access element using [{#,#,..,#}]
	void get_pos(unsigned int *const pos, unsigned int index) const;
	unsigned int get_index(unsigned int *const pos) const;
	
	iterator begin() const { iterator it(this, 0); return it; }
	iterator end() const { iterator it(this, size); return it; }
	
	friend class sub_NDArray<T>;
};


/** **************************************************************************************************************************************
// NDArray<T>::iterator class prototype
/****************************************************************************************************************************************/

template<class T>
class NDArray<T>::iterator {
	T *p;
	NDArray<T> *parent;
	unsigned int *pos;	// Gives the current position in each dimension of the iterator
	unsigned int N;
public:
	// Constructors
	iterator();
	iterator(T *_p, unsigned int _N);
	iterator(const NDArray<T> *const _parent, unsigned int index);
	iterator(const iterator &it);
	~iterator();
	
	// Operators
	iterator& operator ++();
	iterator& operator --();
	iterator& operator +=(const int &rhs);
	iterator& operator -=(const int &rhs);
	iterator& operator +(const int &rhs);
	iterator& operator -(const int &rhs);
	
	bool operator ==(const iterator &rhs) { return (p == rhs.p); }	// Comparison
	bool operator !=(const iterator &rhs) { return (p != rhs.p); }
	bool operator <(const iterator &rhs) { return (p < rhs.p); }
	bool operator <=(const iterator &rhs) { return (p <= rhs.p); }
	bool operator >(const iterator &rhs) { return (p > rhs.p); }
	bool operator >=(const iterator &rhs) { return (p >= rhs.p); }
	void operator =(const iterator &rhs);
	
	T& operator *() { return *p; }					// Dereference
	
	// Mutators
	void set_parent(NDArray<T> &_parent) { parent = &_parent; parent->get_pos(pos, get_offset()); N = parent->N; }
	void set_pos(unsigned int index, unsigned int _pos_i) { pos[index] = _pos_i; }
	
	// Accessors
	void get_pos(unsigned int *_pos) { for(unsigned int i=0; i<N; i++) { _pos[i] = pos[i]; } }
	const unsigned int &get_pos(unsigned int index) { return pos[index]; }
	unsigned int get_offset();
};

/** **************************************************************************************************************************************
// NDArray<T>::iterator class members
/****************************************************************************************************************************************/

template<class T>
NDArray<T>::iterator::iterator() : p(NULL), parent(NULL), pos(NULL) { N = 0; }

template<class T>
NDArray<T>::iterator::iterator(T *_p, unsigned int _N) : p(_p), parent(NULL), pos(NULL), N(_N) { pos = new unsigned int[N]; }

template<class T>
NDArray<T>::iterator::iterator(const NDArray<T> *const _parent, unsigned int index) : p(NULL), parent(NULL), pos(NULL) {
	parent = (NDArray<T>*)_parent;
	N = parent->N;
	pos = new unsigned int[N];
	p = &(parent->element[index]);
	parent->get_pos(pos, index);
}

template<class T>
NDArray<T>::iterator::iterator(const iterator &it) : p(NULL) , parent(NULL), pos(NULL) {
	N = it.N;
	p = it.p;
	parent = it.parent;
	pos = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) { pos[i] = it.pos[i]; }
}

template<class T>
NDArray<T>::iterator::~iterator() { if(pos != NULL) { delete[] pos; } }

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator ++() {					// Increment one
	++p;
	if(parent != NULL) {
		pos[N-1]++;
		for(unsigned int i=N-1; i>0; i--) {
			if(pos[i] >= parent->get_width(i)) {
				pos[i] = 0;
				pos[i-1] += 1;
			} else {
				return *this;
			}
		}
	}
	return *this;
}

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator --() {					// Decrement one
	--p;
	if(parent != NULL) {
		for(unsigned int i=N-1; i>0; i--) {
			if(pos[i] != 0) {
				pos[i]--;
				return *this;
			} else {
				pos[i] = parent->get_width(i) - 1;
			}
		}
		if(pos[0] != 0) { pos[0]--; } else { p = NULL; }
	}
	return *this;
}

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator +=(const int &rhs) {
	p += rhs;
	if(parent != NULL) { parent->get_pos(pos, get_offset()); }
	return *this;
}

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator -=(const int &rhs) {
	p -= rhs;
	if(parent != NULL) { parent->get_pos(pos, get_offset()); }
	return *this;
}

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator +(const int &rhs) {
	p += rhs;
	if(parent != NULL) { parent->get_pos(pos, get_offset()); }
	return *this;
}

template<class T>
typename NDArray<T>::iterator& NDArray<T>::iterator::operator -(const int &rhs) {
	p -= rhs;
	if(parent != NULL) { parent->get_pos(pos, get_offset()); }
	return *this;
}

template<class T>
void NDArray<T>::iterator::operator =(const iterator &rhs) {	// Assignment
	if(N != rhs.N){
		N = rhs.N;
		pos = new unsigned int[N];
	}
	p = rhs.p;
	parent = rhs.parent;
	for(unsigned int i=0; i<N; i++) { pos[i] = rhs.pos[i]; }
}

template<class T>
unsigned int NDArray<T>::iterator::get_offset() {
	if(parent == NULL) { return 0; }
	return p - parent->begin().p;
}


/** **************************************************************************************************************************************
// NDArray<T> class members
/****************************************************************************************************************************************/

template<class T>
NDArray<T>::NDArray(unsigned int _N) : element(NULL), N(_N), width(NULL), multiplier(NULL) {
	assert(N != 0);
	size = 1;
	width = new unsigned int[N];
	multiplier = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) {
		width[i] = 0;
		multiplier[i] = 1;
	}
	element = new T[size];
}

template<class T>
NDArray<T>::NDArray(unsigned int *_width, unsigned int _N) : element(NULL), N(_N), width(NULL), multiplier(NULL) {
	assert(N != 0);
	width = new unsigned int[N];
	size = 1;
	for(unsigned int i=0; i<N; i++) {
		width[i] = _width[i];
		size *= width[i];
	}
	multiplier = new unsigned int[N];
	multiplier[N-1] = 1;
	for(unsigned int i=N-1; i>0; i--) { multiplier[i-1] = multiplier[i]*width[i]; }
	element = new T[size];
}

template<class T>
NDArray<T>::NDArray(const NDArray<T> &original) : element(NULL), width(NULL), multiplier(NULL) {
	N = original.N;
	size = original.size;
	width = new unsigned int[N];
	multiplier = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) {
		width[i] = original.width[i];
		multiplier[i] = original.multiplier[i];
	}
	element = new T[size];
	for(unsigned int i=0; i<size; i++) { element[i] = original.element[i]; }
}

template<class T>
NDArray<T>::NDArray(const NDArray<T> &original, const unsigned int *const flatten_axes, unsigned int N_axes) : element(NULL), width(NULL), multiplier(NULL) {
	assert(original.N > N_axes);
	N = original.N - N_axes;
	
	// Determine which axes to use
	unsigned int *original_index = new unsigned int[N];
	unsigned int k = 0;
	for(unsigned int i=0; i<original.N; i++) {
		if(flatten_axes[i-k] != i) {
			assert(k < N);
			original_index[k] = i;
			k++;
		}
	}
	assert(k == N);
	
	// Determine size, width and multiplier
	size = 1;
	width = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) {
		width[i] = original.width[original_index[i]];
		size *= width[i];
	}
	multiplier = new unsigned int[N];
	multiplier[N-1] = 1;
	for(unsigned int i=N-1; i>0; i--) { multiplier[i-1] = multiplier[i]*width[i]; }
	
	// Initialize array to zero
	bool *initialized = new bool[size];
	element = new T[size];
	for(unsigned int i=0; i<size; i++) { initialized[i] = false; }
	
	// Fill new array
	unsigned int *_pos = new unsigned int[original.N];
	iterator it = original.begin();
	iterator it_end = original.end();
	unsigned int tmp_i;
	for(; it != it_end; ++it) {
		it.get_pos(_pos);
		for(unsigned int k=0; k<N; k++) {
			_pos[k] = _pos[original_index[k]];
		}
		tmp_i = get_index(_pos);
		if(initialized[tmp_i]){ element[tmp_i] += *it; } else { initialized[tmp_i] = true; element[tmp_i] = *it; }
	}
	
	delete[] initialized;
	delete[] _pos;
	delete[] original_index;
}

template<class T>
NDArray<T>::NDArray(const sub_NDArray<T> &original) : element(NULL), width(NULL), multiplier(NULL)  {
	N = original.N;
	size = 1;
	width = new unsigned int[N];
	multiplier = new unsigned int[N];
	for(unsigned int i=0; i<N; i++) {
		width[i] = original.width[i];
		multiplier[i] = original.multiplier[i];
		size *= width[i];
	}
	element = new T[size];
	for(unsigned int i=0; i<size; i++) { element[i] = original.element[i]; }
}

template<class T>
NDArray<T>::~NDArray() {
	delete[] element;
	delete[] width;
	delete[] multiplier;
}

template<class T>
void NDArray<T>::resize(const unsigned int *const _width, unsigned int _N) {
	if(_N != N) {
		delete[] width;
		delete[] multiplier;
		N = _N;
		width = new unsigned int[N];
		multiplier = new unsigned int[N];
	}
	unsigned int old_size = size;
	size = 1;
	for(unsigned int i=0; i<N; i++) {
		width[i] = _width[i];
		size *= width[i];
	}
	if(size != old_size) {
		delete[] element;
		element = new T[size];
	}
	multiplier[N-1] = 1;
	for(unsigned int i=N-1; i>0; i--) { multiplier[i-1] = multiplier[i]*width[i]; }
	element = new T[size];
}

template<class T>
sub_NDArray<T> NDArray<T>::operator [](unsigned int index) {
	assert(index < width[0]);
	sub_NDArray<T> sub_arr(&element[index*multiplier[0]], &width[1], &multiplier[1], N-1);
	return sub_arr;
}

template<class T>
NDArray<T>& NDArray<T>::operator =(const NDArray<T> &rhs) {
	if(this == &rhs) { return *this; }
	if(rhs.N != N) {
		delete[] width;
		delete[] multiplier;
		N = rhs.N;
		width = new unsigned int[N];
		multiplier = new unsigned int[N];
	}
	unsigned int rhs_size = 1;
	for(unsigned int i=0; i<N; i++) {
		multiplier[i] = rhs.multiplier[i];
		width[i] = rhs.width[i];
		rhs_size *= rhs.width[i];
	}
	if(size != rhs_size) {
		size = rhs_size;
		delete[] element;
		element = new T[size];
	}
	for(unsigned int i=0; i<size; i++) { element[i] = rhs.element[i]; }
	return *this;
}

template<class T>
NDArray<T>& NDArray<T>::operator =(const sub_NDArray<T> &rhs) {
	if(rhs.N != N) {
		delete[] width;
		delete[] multiplier;
		N = rhs.N;
		width = new unsigned int[N];
		multiplier = new unsigned int[N];
	}
	unsigned int rhs_size = 1;
	for(unsigned int i=0; i<N; i++) {
		multiplier[i] = rhs.multiplier[i];
		width[i] = rhs.width[i];
		rhs_size *= rhs.width[i];
	}
	if(size != rhs_size) {
		size = rhs_size;
		delete[] element;
		element = new T[size];
	}
	for(unsigned int i=0; i<size; i++) { element[i] = rhs.element[i]; }
	return *this;
}

template<class T>
bool NDArray<T>::operator ==(const NDArray<T>& rhs) const {
	if(this == &rhs) { return true; }
	if(rhs.N != N) { return true; }
	if(size != rhs.size) { return false; }
	for(unsigned int i=0; i<N; i++) {
		if(width[i] != rhs.width[i]) { return false; }
	}
	for(unsigned int i=0; i<size; i++) {
		if(element[i] != rhs.element[i]) { return false; }
	}
	return true;
}

template<class T>
bool NDArray<T>::operator ==(const sub_NDArray<T>& rhs) const {
	if(rhs.N != N) { return false; }
	for(unsigned int i=0; i<N; i++) {
		if(width[i] != rhs.width[i]) { return false; }
	}
	for(unsigned int i=0; i<size; i++) {
		if(element[i] != rhs.element[i]) { return false; }
	}
	return true;
}

template<class T>
void NDArray<T>::get_pos(unsigned int *const pos, unsigned int index) const {
	for(unsigned int i=N-1; i>0; i--) {
		pos[i] = (index % width[i]);
		index = (index - pos[i]) / width[i];
	}
	pos[0] = index;
}

template<class T>
unsigned int NDArray<T>::get_index(unsigned int *pos) const {
	unsigned int index = 0;
	for(unsigned int i=0; i<N; i++) {
		index += pos[i] * multiplier[i];
	}
	return index;
}

template<class T>
T& NDArray<T>::get_element(const unsigned int *const index, unsigned int _N) const {
	assert(N == _N);
	unsigned int flat_index = 0;
	for(unsigned int i=0; i<N; i++) {
		assert(index[i] < width[i]);
		flat_index += multiplier[i]*index[i];
	}
	return element[flat_index];
}


template<class T>
template< class TParams>
void NDArray<T>::fill(T (*func)(const unsigned int *const pos, unsigned int _N, const TParams &params), const TParams &params) {	// Fill the array with a function of position
	unsigned int pos[N];
	iterator it_end = end();
	for(iterator it=begin(); it!=it_end; ++it) {
		it.get_pos(&pos[0]);
		*it = func(&pos[0], N, params);
	}
}

template<class T>
void NDArray<T>::fill(T (*func)(const unsigned int *const pos, unsigned int _N)) {
	iterator it = begin();
	iterator it_end = end();
	for(; it != it_end; ++it) { *it = func(it.get_pos(), N); }
}


#endif // _NDARRAY_H__