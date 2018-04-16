
.DEFAULT: helloworld
.PHONY: all

all:	helloworld

helloworld:	helloworld.cc
	g++ helloworld.cc -o helloworld -l opencv_core -l opencv_imgproc -l opencv_highgui -l glog -l ceres -I /usr/local/include/eigen3 

clean:
	rm helloworld

