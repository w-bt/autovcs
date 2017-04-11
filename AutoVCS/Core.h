#pragma once
#include "Share.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <io.h>
#include <time.h>
#include <string>

using namespace cv;
using namespace std;

class Core
{
public:
	int 	widthSI, heightSI, widthCanvas, heightCanvas, widthIA, heightIA;
	Mat 	SI, filteredSI, canvas;
	static	int optScheme;
	double	duration1run;
	string	ext, singlePath, singleSI, sharePath;
	vector<string> siPath, siName, sharesPath, ssSize;
	vector<double> dur1, dur2;

	Core();
	void	reset();
	int 	viewMainMenu();
	void 	viewEncoding();
	void 	viewDecoding();
	void 	centerString(char* s);
	void	chooseScheme();
	void	viewEncodingIMSVCS(Share& S1, Share& S2);
	void	viewEncodingNIMSVCS(Share& S1, Share& S2);
	void 	setSecretImagePath();
	void 	setSecretImage(string path);
	void 	filter();
	void	generateShareIMSVCS(Share& S1, Share& S2);
	Mat		generateNewImg(int h, int w, int channel);
	void	setSharePath();
	void	viewDecodingIMSVCS();
	void	viewDecodingNIMSVCS();
	void	viewDecodingNIMSVCS2();
	void	getIntersectionSize(Share& S1, Share& S2);
	void	getNormalPointStart(Share& S1, Share& S2);
	void	stackingIMSVCS(Share& S1, Share& S2);
	void	splitString(const string &s, char delim, vector<string> &elems);
	vector<string> splitString(const string &s, char delim);
	void	setShadowSharePath(Share& S1, Share& S2);
	void	setSSSize(Share& S1, Share& S2, string size);
	void	setSecretImageSize(string size);
	static int getRandomNumber();
	void	genIntersection(Share& S1, Share& S2);
	void	randomSIPosition(Share& S1, Share& S2);
	void	generateShareNIMSVCS(Share& S1, Share& S2);
	void	getNormalRotationImage(Share& S);
	long long int	getPossibilities(Share S1, Share S2);
	void	estimate1TimeRun(Share S1, Share S2);
	void	estimateCanvas(Share& S1, Share& S2, int r);
	void	estimateStack(Share& S1, Share& S2, int r);
	void	manualStacking(Share& S1, Share& S2, string path);
	void	getRotationImage(Share& S);
	int		getIndexEstimation(Share& S1, Share& S2);
};

