default:
	python setup.py build_ext -b .

build:
	cython -v -t --cplus pyanimats/c_animat/c_animat.pyx

clean: 
	rm -r build
	rm pyanimats/c_animat*.so

test: default
	py.test test
