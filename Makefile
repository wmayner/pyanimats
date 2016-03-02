default:
	python setup.py build_ext --inplace

build:
	cython -v -t --cplus c_animat/c_animat.pyx

clean: 
	rm -r build
	rm c_animat*.so

test: default
	py.test test
