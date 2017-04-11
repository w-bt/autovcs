#include "stdafx.h"
#include "targetver.h"
#include "Share.h"
#include "Core.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <set>
#include <algorithm>    // std::shuffle, sort
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <iostream>     // std::cout
#include <vector>       // std::vector
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace std;

Share::Share(int i)
{
	index		= i;
	angle		= 0;
	revAngle	= 0;
	widthSS 	= 0;
	heightSS 	= 0;
	width		= 0;
	height		= 0;
	rotWidth	= 0;
	rotHeight	= 0;
}


Share::~Share(void)
{
}

void Share::setShare(string path){
	vector<Mat> tempChannels(4);
	int flagS	= 0;
	share		= imread(path + "share" + to_string(index) + ".png", IMREAD_UNCHANGED);
	if(share.data > 0){
		split(share,tempChannels);
		channels	= tempChannels;
		width 		= share.cols;
		height 		= share.rows;
		flagS		= 1;
	} else {
		cout << "\nNo share found" << endl;
	}
}

void Share::setAvailableOrientation(int widthSI, int heightSI){
	/*if((widthSS >= widthSI)&&(widthSS >= heightSI)&&(heightSS >= widthSI)&&(heightSS >= heightSI)){
		availableOrientation.clear();
		availableOrientation.push_back(0);
		availableOrientation.push_back(1);
		availableOrientation.push_back(2);
		availableOrientation.push_back(3);
	} else {
		if((widthSS >= widthSI)&&(heightSS >= heightSI)){
			availableOrientation.clear();
			availableOrientation.push_back(0);
			availableOrientation.push_back(2);
		} else {
			availableOrientation.clear();
			availableOrientation.push_back(1);
			availableOrientation.push_back(3);
		}
	}*/
	availableOrientation.clear();
	availableOrientation.push_back(0);
	availableOrientation.push_back(2);
}

void Share::genRandomOrientation(){
	/*int size, r, c;

	size	= availableOrientation.size();
	r		= Core::getRandomNumber();
	c		= r % size;

	if(availableOrientation[c] == 0){
		angle		= 0;
	} else {
		if(availableOrientation[c] == 1){
			angle		= 90;
		} else {
			if(availableOrientation[c] == 2){
				angle		= 180;
			} else {
				if(availableOrientation[c] == 3){
					angle		= 270;
				}
			}
		}
	}

	revAngle	= (360 - angle) % 360;*/
	angle			= 0;
	revAngle		= 0;
}

void Share::getRotationSize(){
	if((angle == 0)||(angle == 180)){
		if(share.data > 0){
			rotWidth	= width;
			rotHeight	= height;
		} else {
			rotWidthSS	= widthSS;
			rotHeightSS	= heightSS;
		}
	} else {
		if(share.data > 0){
			rotWidth	= height;
			rotHeight	= width;
		} else {
			rotWidthSS	= heightSS;
			rotHeightSS	= widthSS;
		}
	}
}

void Share::setIntersection(int startX, int endX, int startY, int endY){
	intersection.clear();
	intersection.push_back(Point(startX,startY));
	intersection.push_back(Point(endX,startY));
	intersection.push_back(Point(startX,endY));
	intersection.push_back(Point(endX,endY));
}