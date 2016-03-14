default:
	python setup.py build_ext -b .

build:
	cython -v -t --cplus pyanimats/c_animat/c_animat.pyx

clean: 
	rm -rf build
	rm -f pyanimats/c_animat*.so

test: default
	py.test test
