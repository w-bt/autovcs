#include "stdafx.h"
#include "Core.h"
#include "Share.h"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <direct.h>
#include <time.h>
#include <io.h>
#include <set>
#include <algorithm>    // std::shuffle, sort
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <thread>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>

#if defined(_MSC_VER) && _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4146 4244 4800)
#endif
#include <mpirxx.h>
#if defined(_MSC_VER) && _MSC_VER
#pragma warning(pop)
#endif

using namespace cv;
using namespace std;

int M0[2][4] = {
		{255, 255, 0, 0},
		{255, 255, 0, 0}
	};

int M1[2][4] = {
		{255, 255, 0, 0},
		{0, 0, 255, 255}
	};

array<int,4> pos = {0,1,2,3};

int Core::optScheme;

Core::Core()
{
	reset();

	_mkdir("result/");
	_mkdir("result/encoding/");
	_mkdir("result/encoding/imsvcs/");
	_mkdir("result/encoding/nimsvcs/");
	_mkdir("result/decoding/");
	_mkdir("result/decoding/imsvcs/");
	_mkdir("result/decoding/nimsvcs/");
}

void Core::reset(){
	widthSI		= 0;
	heightSI	= 0;
	widthCanvas = 0;
	heightCanvas= 0;
	SI.release();
	filteredSI.release();
	canvas.release();
}

int Core::viewMainMenu(){
	system("cls");
	int ans = 0;

	cout << "================================================================================" << endl;
	cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << endl;
	cout << "================================================================================" << endl;
	cout << "\n";
	centerString("Option :");
	cout << "\n\n" << endl;
	centerString("(1) Encoding");
	cout << "\n" << endl;
	centerString("(2) Decoding");
	cout << "\n" << endl;
	centerString("(3) Exit");
	cout << "\n\n" << endl;
	cout << "================================================================================" << endl;
	cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << endl;
	cout << "================================================================================" << endl;
	while((ans != 1)&&(ans != 2)&&(ans != 3)){
		cout << "\n> Your choice (1/2/3) ? ";
		cin >> ans;
		cin.ignore (std::numeric_limits<std::streamsize>::max(), '\n');
	}

	return ans;
}

void Core::centerString(char* s)
{
	int l 	= strlen(s);
   	int pos = (int)((80-l)/2);
   	for(int i = 0; i < pos; i++)
    	cout << " ";

   	cout << s;
}

void Core::viewEncoding(){
	system("cls");
	// *************************************************************************************
	// ******************************** PRE-DECOMPOSING ************************************
	// *************************************************************************************
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "*******************************************************************************" << endl;
	cout << "***************************** PRE-DECOMPOSING *********************************" << endl;
	cout << "*******************************************************************************" << endl;
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	// =====================================================================================
	// ================================= DETERMINING SCHEME ================================
	// =====================================================================================
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "\n===============================================================================" << endl;
	cout << "============================= DETERMINING SCHEME ==============================" << endl;
	cout << "===============================================================================" << endl;
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	chooseScheme();
	Share S1(1);
	Share S2(2);
	if(optScheme == 1){
		viewEncodingIMSVCS(S1,S2);
	} else {
		if(optScheme == 2){
			viewEncodingNIMSVCS(S1,S2);
		}
	}
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "\n> Finished" << endl;
	cout << "\n" ;
	system("pause");
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	// =====================================================================================
	int opt = viewMainMenu();
	S1.~Share();
	S2.~Share();
	reset();
	if(opt == 1){
		viewEncoding();
	} else {
		if(opt == 2){
			viewDecoding();
		} else {
			exit(0);
		}
	}
}

void Core::chooseScheme(){
	int flag	= 0;
	int ans;
	while(flag == 0){
		cout << "\n> Choose scheme (1) IMSVCS (2) NIMSVCS : ";
		cin >> ans;
		if(ans == 1){
			flag		= 1;
			optScheme	= 1;
			sharePath	= "result/encoding/imsvcs/";
		} else {
			if(ans == 2){
				flag		= 1;
				optScheme	= 2;
				sharePath	= "result/encoding/nimsvcs/";
			} else {
				cout << "\nPlease input (1) or (2)";
			}
		}
	}
}

void Core::viewEncodingIMSVCS(Share& S1, Share& S2){
	// =====================================================================================
	// ============================== DETERMINING SECRET IMAGE =============================
	// =====================================================================================
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "\n===============================================================================" << endl;
	cout << "========================== DETERMINING SECRET IMAGE ===========================" << endl;
	cout << "===============================================================================" << endl;
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	setSecretImagePath();
	Mat tempSI;
	fstream file;
	file.open("result/encoding/imsvcs/durdecode.csv",fstream::out);
	for(int i = 0; i < siPath.size(); i++){

		singlePath	= siPath[i];
		singleSI	= siName[i];
		setSecretImage(singlePath);
		cout << "\nSI's size = " << widthSI << "x" << heightSI << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// *************************************************************************************
		// *********************************** DECOMPOSING *************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "******************************** DECOMPOSING **********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	
		// =====================================================================================
		// =============================== FILTERING SECRET IMAGE ==============================
		// =====================================================================================
		// filtering secret image
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "========================== FILTERING SECRET IMAGE =============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Filtering secret image" << endl;
		filter();
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// =====================================================================================
		// ================================ GENERATING SHARE =================================
		// =====================================================================================
		// generating share
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "============================== GENERATING SHARE ===============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Generating shares" << endl;

		clock_t tStart	= clock();
		generateShareIMSVCS(S1, S2);
		double duration	= (double)(clock() - tStart)/CLOCKS_PER_SEC;
		file << (i+1) << "," << fixed << setprecision(5) << duration << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
	}
	file.close();

	file.open(sharePath +  "info.txt",fstream::out);
	for(int i = 0; i < siPath.size(); i++){
		file << sharePath << siName[i] << "/" << endl;
	}
	file.close();
}

void Core::setSecretImagePath(){
	string pathSI;
	cin.ignore(); 
	cout << "\n> Input SI's path : ";
	getline (cin, pathSI);

    ifstream file;
	file.open(pathSI);
	while(!file){
		cout << "\nNo file found" << endl;
		cout << "\n> Input SI's path : ";
		cin >> pathSI;
		file.open(pathSI);
	}
	vector<string> tempPath = splitString(pathSI,'/');
	string basePath 		= "";
	for(int i = 0; i < (tempPath.size() - 1); i++){
		basePath = basePath + tempPath[i] + "/";
	}
    string str;
	int no	= 1;
	siPath.clear();
	siName.clear();
    while (getline(file, str))
    {
    	siPath.push_back(basePath + str);
		siName.push_back(str);
    }
}

void Core::setSecretImage(string pathSI)
{
	Mat tempSI;
	ext 		= pathSI.substr(pathSI.size() - 3);
	tempSI		= imread(pathSI,0);
	SI			= tempSI;
	widthSI		= tempSI.cols;
	heightSI	= tempSI.rows;
	cout << "\nSecret Image's size : " << heightSI << "x" << widthSI << endl;
}

void Core::filter(){
	Mat imgTemp = SI.clone();
	if(ext != "bmp"){
		// histogram equalization
		equalizeHist(SI, imgTemp);
	
		// image sharpening (laplacian)
		Mat sharp;
		Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
		filter2D(imgTemp, sharp, imgTemp.depth(), kernel);

		// dithering floyd
		for (int r = 0; r < sharp.rows; r++) {
			for (int c = 0; c < sharp.cols; c++) {
				uchar oldPixel	= sharp.at<uchar>(r,c);
				uchar newPixel	= oldPixel > 127 ? 255 : 0;
				sharp.at<uchar>(r,c)	= newPixel;
				int quantError			= oldPixel - newPixel;
				if(c+1 < sharp.cols) sharp.at<uchar>(r,c+1)					+=  (7 * quantError) >> 4;
				if((c > 0)&&(r+1 < sharp.rows)) sharp.at<uchar>(r+1,c-1)	+=  (3 * quantError) >> 4;
				if(r+1 < sharp.rows) sharp.at<uchar>(r+1,c)					+=  (5 * quantError) >> 4;
				if((r+1 < sharp.rows)&&(c+1 < sharp.cols)) sharp.at<uchar>(r+1,c+1)	+=  (1 * quantError) >> 4;
			}
		}
	}

	filteredSI = imgTemp;
}

void Core::generateShareIMSVCS(Share& S1, Share& S2){
	Mat C1			= generateNewImg(heightSI*2, widthSI*2, 1);
	Mat A1			= Mat::zeros(heightSI*2, widthSI*2, CV_8UC1);
	Mat C2			= generateNewImg(heightSI*2, widthSI*2, 1);
	Mat A2			= Mat::zeros(heightSI*2, widthSI*2, CV_8UC1);

	vector<Mat> channels1, channels2;
	unsigned seed;

	for(int i = 0; i < heightSI; i++){
		for(int j = 0; j < widthSI; j++){
			// obtain a time-based seed:
			// random permutation on 'position' to get matrix c
			seed = (chrono::system_clock::now().time_since_epoch().count())+j;
			shuffle (pos.begin(), pos.end(), default_random_engine(seed));
			int C[2][4];
			// permasalahan ada di sini
			if(filteredSI.at<uchar>(i,j) < 128){
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M1[u][v];
					}
				}
			} else {
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M0[u][v];
					}
				}
			}
			// transform 1D to 2D matrix m
			int m_1[2][2],m_2[2][2];
			for(int l = 0; l < 4; l++){
				int rowm		= l/2;
				m_1[rowm][l % 2]	= C[0][pos[l]];
				m_2[rowm][l % 2]	= C[1][pos[l]];
			}
			// asign matrix m to the share
			int marky = 0;
			for(int newy = i*2; newy < (i*2)+2; newy++){
				int markx = 0;
				for(int newx = j*2; newx < (j*2)+2; newx++){
					C1.at<uchar>(newy,newx) = m_1[marky][markx];
					if(m_1[marky][markx] == 255){
						A1.at<uchar>(newy,newx) = 0;
					} else {
						A1.at<uchar>(newy,newx) = 255;
					}
					
					C2.at<uchar>(newy,newx) = m_2[marky][markx];
					if(m_2[marky][markx] == 255){
						A2.at<uchar>(newy,newx) = 0;
					} else {
						A2.at<uchar>(newy,newx) = 255;
					}
					markx = markx + 1;
				}
				marky = marky + 1;
			}
		}
	}

	string foldername	= "result/encoding/imsvcs/" + singleSI;
	_mkdir(foldername.c_str());

	channels1.push_back(C1);
	channels1.push_back(C1);
	channels1.push_back(C1);
	channels1.push_back(A1);

	channels2.push_back(C2);
	channels2.push_back(C2);
	channels2.push_back(C2);
	channels2.push_back(A2);

	merge(channels1, S1.share);
	merge(channels2, S2.share);

	imwrite(foldername + "/share1.png",S1.share);
	imwrite(foldername + "/share2.png",S2.share);
}

Mat Core::generateNewImg(int h, int w, int channel){
	Mat newImage;
	if(channel == 1){
		newImage.create(h, w, CV_8UC1);
		newImage = Scalar(255);
	} else {
		if(channel == 3){
			newImage.create(h, w, CV_8UC3);
			newImage = Scalar(255,255,255);
		} else {
			if(channel == 4){
				newImage.create(h, w, CV_8UC4);
				newImage = Scalar(255,255,255,0);
			}
		}
	}
	return newImage;
}

void Core::viewDecoding(){
	system("cls");
	chooseScheme();
	if(optScheme == 1){
		viewDecodingIMSVCS();
	} else {
		if(optScheme == 2){
			//viewDecodingNIMSVCS();
			viewDecodingNIMSVCS2();
		}
	}
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "\n> Finished" << endl;
	cout << "\n" ;
	system("pause");
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	// =====================================================================================
	int opt = viewMainMenu();
	reset();
	if(opt == 1){
		viewEncoding();
	} else {
		if(opt == 2){
			viewDecoding();
		} else {
			exit(0);
		}
	}
}

void Core::setSharePath(){
	char pathShare[100];
	cout << "\n> Input Share's path : ";
	cin >> pathShare;
	ifstream file;
	file.open(pathShare);
	while(!file){
		cout << "\nNo file found" << endl;
		cout << "\n> Input Share's path : ";
		cin >> pathShare;
		file.open(pathShare);
	}
    string str;
	int no	= 1;
    while (getline(file, str))
    {
		sharesPath.push_back(str);
    }
}

void Core::viewDecodingIMSVCS(){
	setSharePath();
	fstream file;
	file.open("result/decoding/imsvcs/durdecode.csv",fstream::out);
	for(int i = 0; i < sharesPath.size(); i++){
		vector<string> tempName = splitString(sharesPath[i], '/');
		singleSI	= tempName[tempName.size() - 1];
		// *************************************************************************************
		// ********************************* PRE-DECODING **************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "****************************** PRE-DECODING ***********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Input share 1 and share 2 " << endl;
		Share S1(1);
		S1.setShare(sharesPath[i]);
		cout << "\nS1's size = " << S1.width << "x" << S1.height << endl;

		Share S2(2);
		S2.setShare(sharesPath[i]);
		cout << "\nS2's size = " << S2.width << "x" << S2.height << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// *************************************************************************************
		// ************************************ STACKING ***************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "********************************* STACKING ************************************" << endl;
		cout << "*******************************************************************************" << endl;
		cout << "\n> Stacking share 1 and share 2 " << endl;

		clock_t tStart	= clock();
		stackingIMSVCS(S1, S2);
		double duration	= (double)(clock() - tStart)/CLOCKS_PER_SEC;
		file << (i+1) << "," << fixed << setprecision(5) << duration << endl;

		string newFolder	= "result/decoding/imsvcs/" + singleSI + "/";
		_mkdir(newFolder.c_str());
		imwrite("result/decoding/imsvcs/" + singleSI + "/ri.png",canvas);
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
		S1.~Share();
		S2.~Share();
	}
	file.close();
}

void Core::stackingIMSVCS(Share& S1, Share& S2){
	vector<Mat> channelsBB;
	
	for(int i = 0; i < S1.height; i++){
		for(int j = 0; j < S1.width; j++){
			int currentP	= S2.channels[0].at<uchar>(i,j);
			if(currentP < 128){
				S1.channels[0].at<uchar>(i,j)	= S2.channels[0].at<uchar>(i,j);
				S1.channels[3].at<uchar>(i,j)	= 255;
			}
		}
	}
	
	channelsBB.push_back(S1.channels[0]);
    channelsBB.push_back(S1.channels[0]);
    channelsBB.push_back(S1.channels[0]);
	channelsBB.push_back(S1.channels[3]);

	merge(channelsBB, canvas);
}

void Core::splitString(const string &s, char delim, vector<string> &elems) {
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

vector<string> Core::splitString(const string &s, char delim) {
    vector<string> elems;
    splitString(s, delim, elems);
    return elems;
}

void Core::viewEncodingNIMSVCS(Share& S1, Share& S2){
	// =====================================================================================
	// =============================== DETERMINING SS SIZE =================================
	// =====================================================================================
	// determining the size of shadow share
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	cout << "\n===============================================================================" << endl;
	cout << "========================== DETERMINING SS SIZE ================================" << endl;
	cout << "===============================================================================" << endl;
	// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	setShadowSharePath(S1, S2);
	setSecretImagePath();
	string type;
	for(int i = 0; i < ssSize.size(); i++){
		dur1.push_back(0);
		dur2.push_back(0);

		setSSSize(S1,S2,ssSize[i]);
		if(siPath.size() == 1){
			singlePath	= siPath[0];
			type		= "1";
		} else {
			singlePath	= siPath[i];
			type		= "2";
		}
		setSecretImage(singlePath);

		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		S2.setAvailableOrientation(widthSI, heightSI);
	
		if(S2.availableOrientation.size() == 4){
			cout << "\nSS2's available orientation = " << S2.availableOrientation[0] << ", " << S2.availableOrientation[1] << ", " << S2.availableOrientation[2] << ", and " << S2.availableOrientation[3] << endl;
		} else {
			cout << "\nSS2's available orientation = " << S2.availableOrientation[0] << " and " << S2.availableOrientation[1] << endl;
		}
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
	
		// =====================================================================================
		// ========================== CALCULATING INTERSECTION AREA ============================
		// =====================================================================================
		// calculating intersection area
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "====================== CALCULATING INTERSECTION AREA ==========================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Choosing random angle rotation" << endl;
		S2.genRandomOrientation();
	
		cout << "\nSS2's angle orientation = " << S2.angle << endl;

		S2.getRotationSize();
		cout << "\nSS2's rotated width = " << S2.rotWidthSS << endl;
		cout << "\nSS2's rotated height = " << S2.rotHeightSS << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Calculating intersection area" << endl;

		genIntersection(S1,S2);

		cout << "\nIntersection size " << (S1.intersection[2].y - S1.intersection[0].y) << " x " << (S1.intersection[1].x - S1.intersection[0].x) << endl;
		cout << "\nIntersection area for SS1 : " << endl;
		cout << "\t[1] (" << S1.intersection[0].x << "," << S1.intersection[0].y << ")" << endl;
		cout << "\t[2] (" << S1.intersection[1].x << "," << S1.intersection[1].y << ")" << endl;
		cout << "\t[3] (" << S1.intersection[2].x << "," << S1.intersection[2].y << ")" << endl;
		cout << "\t[4] (" << S1.intersection[3].x << "," << S1.intersection[3].y << ")" << endl;
		cout << "\nIntersection area for rotated SS2 : " << endl;
		cout << "\t[1] (" << S2.intersection[0].x << "," << S2.intersection[0].y << ")" << endl;
		cout << "\t[2] (" << S2.intersection[1].x << "," << S2.intersection[1].y << ")" << endl;
		cout << "\t[3] (" << S2.intersection[2].x << "," << S2.intersection[2].y << ")" << endl;
		cout << "\t[4] (" << S2.intersection[3].x << "," << S2.intersection[3].y << ")" << endl;

		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// =====================================================================================
		// ============================ GENERATE RANDOM SI POSITION ============================
		// =====================================================================================
		// generating random SI position
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "======================== GENERATE RANDOM SI POSITION ==========================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Generate random SI position" << endl;
	
		randomSIPosition(S1, S2);
	
		cout << "\nSI position in SS1:" << endl;
		cout << "\t[A] (" << S1.position[0].x << "," << S1.position[0].y << ")" << endl;
		cout << "\t[B] (" << S1.position[1].x << "," << S1.position[1].y << ")" << endl;
		cout << "\t[C] (" << S1.position[2].x << "," << S1.position[2].y << ")" << endl;
		cout << "\t[D] (" << S1.position[3].x << "," << S1.position[3].y << ")" << endl;

		cout << "\nSI position in rotated SS2:" << endl;
		cout << "\t[A] (" << S2.position[0].x << "," << S2.position[0].y << ")" << endl;
		cout << "\t[B] (" << S2.position[1].x << "," << S2.position[1].y << ")" << endl;
		cout << "\t[C] (" << S2.position[2].x << "," << S2.position[2].y << ")" << endl;
		cout << "\t[D] (" << S2.position[3].x << "," << S2.position[3].y << ")" << endl;
	
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::	
		// =====================================================================================

		// *************************************************************************************
		// *********************************** DECOMPOSING *************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "******************************** DECOMPOSING **********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
	
		// =====================================================================================
		// =============================== FILTERING SECRET IMAGE ==============================
		// =====================================================================================
		// filtering secret image
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "========================== FILTERING SECRET IMAGE =============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Filtering secret image" << endl;

		clock_t tStart1	= clock();
		filter();
		double duration1	= (double)(clock() - tStart1)/CLOCKS_PER_SEC;
		int size1 = dur1.size();
		dur1[size1-1] = duration1;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
	
		// =====================================================================================
		// ================================ GENERATING SHARE =================================
		// =====================================================================================
		// generating share
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "============================== GENERATING SHARE ===============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Generating shares" << endl;

		clock_t tStart2	= clock();
		generateShareNIMSVCS(S1, S2);
		double duration2	= (double)(clock() - tStart2)/CLOCKS_PER_SEC;
		int size2 = dur2.size();
		dur2[size2-1] = duration2;

		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Rotating share 2 to normal orientation" << endl;
		getNormalRotationImage(S2);
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		string folder	= "result/encoding/nimsvcs/" + type + "/";
		_mkdir(folder.c_str());
		folder			= folder + ssSize[i] + "/";
		_mkdir(folder.c_str());
		imwrite(folder + "share1.png",S1.share);
		imwrite(folder + "share2.png",S2.share);

		S1.share.release();
		S2.share.release();
		S2.rotatedShare.release();

		fstream file;
		file.open(folder + "info.txt",fstream::out);
		file << "name : " << ssSize[i] << endl;
		file << "secret image : " << singlePath << endl;
		file << "\nS2 rotation : " << S2.angle << endl;
		
		file << "\nIntersection size " << (S1.intersection[2].y - S1.intersection[0].y) << " x " << (S1.intersection[1].x - S1.intersection[0].x) << endl;
		file << "\nIntersection area for SS1 : " << endl;
		file << "\t[1] (" << S1.intersection[0].x << "," << S1.intersection[0].y << ")" << endl;
		file << "\t[2] (" << S1.intersection[1].x << "," << S1.intersection[1].y << ")" << endl;
		file << "\t[3] (" << S1.intersection[2].x << "," << S1.intersection[2].y << ")" << endl;
		file << "\t[4] (" << S1.intersection[3].x << "," << S1.intersection[3].y << ")" << endl;
		file << "\nIntersection area for rotated SS2 : " << endl;
		file << "\t[1] (" << S2.intersection[0].x << "," << S2.intersection[0].y << ")" << endl;
		file << "\t[2] (" << S2.intersection[1].x << "," << S2.intersection[1].y << ")" << endl;
		file << "\t[3] (" << S2.intersection[2].x << "," << S2.intersection[2].y << ")" << endl;
		file << "\t[4] (" << S2.intersection[3].x << "," << S2.intersection[3].y << ")" << endl;
		
		file << "\nSI position in SS1:" << endl;
		file << "\t[A] (" << S1.position[0].x << "," << S1.position[0].y << ")" << endl;
		file << "\t[B] (" << S1.position[1].x << "," << S1.position[1].y << ")" << endl;
		file << "\t[C] (" << S1.position[2].x << "," << S1.position[2].y << ")" << endl;
		file << "\t[D] (" << S1.position[3].x << "," << S1.position[3].y << ")" << endl;
		file << "\nSI position in rotated SS2:" << endl;
		file << "\t[A] (" << S2.position[0].x << "," << S2.position[0].y << ")" << endl;
		file << "\t[B] (" << S2.position[1].x << "," << S2.position[1].y << ")" << endl;
		file << "\t[C] (" << S2.position[2].x << "," << S2.position[2].y << ")" << endl;
		file << "\t[D] (" << S2.position[3].x << "," << S2.position[3].y << ")" << endl;
		
		file << "\nS1 position : (" << (S1.pointStart.x * 2) << "," << (S1.pointStart.y * 2) << ")" << endl;
		file << "S2 position : (" << (S2.pointStart.x * 2) << "," << (S2.pointStart.y * 2) << ")" << endl;
		file.close();
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
	}

	fstream file;
	file.open(sharePath + type + "/info.txt",fstream::out);
	for(int i = 0; i < ssSize.size(); i++){
		file << sharePath << type << "/" << ssSize[i] << "/" << endl;
	}
	file.close();

	file.open("result/encoding/nimsvcs/durencode.csv",fstream::out);
	file << "no,filtering,decomposing" << endl;
	for(int i = 0; i < dur1.size(); i++){
		file << (i+1) << "," << fixed << setprecision(5) << dur1[i] << "," << fixed << setprecision(5) << dur2[i] << endl;
	}
	file.close();
	dur1.clear();
	dur2.clear();
}

void Core::setShadowSharePath(Share& S1, Share& S2){
	char pathSShare[100];
	cout << "\n> Input Shadow Share's path : ";
	cin >> pathSShare;
	ifstream file;
	file.open(pathSShare);
	while(!file){
		cout << "\nNo file found" << endl;
		cout << "\n> Input Shadow Share's path : ";
		cin >> pathSShare;
		file.open(pathSShare);
	}
    string str;
	int no	= 1;
	ssSize.clear();
    while (getline(file, str))
    {
		ssSize.push_back(str);
    }
}

void Core::setSSSize(Share& S1, Share& S2, string size){
	vector<string> tempSize = splitString(size, '&');
	vector<string> tempSize1 = splitString(tempSize[0], 'x');
	stringstream(tempSize1[0]) >> S1.heightSS;
	S1.height		= S1.heightSS * 2;
	stringstream(tempSize1[1]) >> S1.widthSS;
	S1.width		= S1.widthSS * 2;
	cout << "\nSS1's size : " << S1.heightSS << "x" << S1.widthSS << endl;
	cout << "\nS1's size : " << S1.height << "x" << S1.width << endl;
	vector<string> tempSize2 = splitString(tempSize[1], 'x');
	stringstream(tempSize2[0]) >> S2.heightSS;
	S2.height		= S2.heightSS * 2;
	stringstream(tempSize2[1]) >> S2.widthSS;
	S2.width		= S2.widthSS * 2;
	cout << "\nSS2's size : " << S2.heightSS << "x" << S2.widthSS << endl;
	cout << "\nS2's size : " << S2.height << "x" << S2.width << endl;
}

int Core::getRandomNumber(){
	int result = rand();

	return result;
}

void Core::genIntersection(Share& S1, Share& S2){
	int widthArea, heightArea, startx, endx, starty, endy, A2X, B2X, A2Y, C2Y, minX, minY, maxX, maxY, avWidthArea, avHeightArea, widthIntersection, heightIntersection;
	vector<Point2i> pos1, pos2;

	widthArea	= (2*S2.rotWidthSS) + (S1.widthSS - (2*widthSI));
	heightArea	= (2*S2.rotHeightSS) + (S1.heightSS - (2*heightSI));

	avWidthArea		= widthArea - S2.rotWidthSS + widthSI;
	avHeightArea	= heightArea - S2.rotHeightSS + heightSI;

	pos1.clear();
	pos1.push_back(Point(S2.rotWidthSS - widthSI, S2.rotHeightSS - heightSI));
	pos1.push_back(Point((S2.rotWidthSS - widthSI) + S1.widthSS, S2.rotHeightSS - heightSI));
	pos1.push_back(Point(S2.rotWidthSS - widthSI, (S2.rotHeightSS - heightSI) + S1.heightSS));
	pos1.push_back(Point((S2.rotWidthSS - widthSI) + S1.widthSS, (S2.rotHeightSS - heightSI) + S1.heightSS));

	int flagW;
	flagW	= 0;
	while(flagW == 0){
		int rx;

		rx	= getRandomNumber();
		//A2X	= rx % avWidthArea;
		A2X	= rx % 30;
		B2X	= A2X + S2.rotWidthSS;

		if((pos1[0].x >= A2X)&&(pos1[0].x < B2X)&&(pos1[1].x >= B2X)){
			startx	= pos1[0].x;
			endx	= B2X;
			minX	= A2X;
			maxX	= pos1[1].x;
		} else {
			if((pos1[0].x >= A2X)&&(pos1[0].x < B2X)&&(pos1[1].x < B2X)){
				startx	= pos1[0].x;
				endx	= pos1[1].x;
				minX	= A2X;
				maxX	= B2X;
			} else {
				if((pos1[0].x <= A2X)&&(pos1[1].x >= B2X)){
					startx	= A2X;
					endx	= B2X;
					minX	= pos1[0].x;
					maxX	= pos1[1].x;
				} else {
					if((pos1[0].x <= A2X)&&(pos1[1].x < B2X)&&(pos1[1].x > A2X)){
						startx	= A2X;
						endx	= pos1[1].x;
						minX	= pos1[0].x;
						maxX	= B2X;
					}
				}
			}
		}

		widthIntersection	= endx - startx;

		if(widthIntersection >= widthSI){
			flagW = 1;
		}
	}

	int flagH;
	flagH	= 0;
	while(flagH == 0){
		int ry;

		ry	= getRandomNumber();
		//A2Y	= ry % avHeightArea;
		A2Y	= 0;
		C2Y	= A2Y + S2.rotHeightSS;

		if((pos1[0].y >= A2Y)&&(pos1[0].y < C2Y)&&(pos1[2].y >= C2Y)){
			starty	= pos1[0].y;
			endy	= C2Y;
			minY	= A2Y;
			maxY	= pos1[2].y;
		} else {
			if((pos1[0].y >= A2Y)&&(pos1[0].y < C2Y)&&(pos1[2].y < C2Y)){
				starty	= pos1[0].y;
				endy	= pos1[2].y;
				minY	= A2Y;
				maxY	= C2Y;
			} else {
				if((pos1[0].y <= A2Y)&&(pos1[2].y >= C2Y)){
					starty	= A2Y;
					endy	= C2Y;
					minY	= pos1[0].y;
					maxY	= pos1[2].y;
				} else {
					if((pos1[0].y <= A2Y)&&(pos1[2].y < C2Y)&&(pos1[2].y > A2Y)){
						starty	= A2Y;
						endy	= pos1[2].y;
						minY	= pos1[0].y;
						maxY	= C2Y;
					}
				}
			}
		}

		heightIntersection	= endy - starty;

		if(heightIntersection >= heightSI){
			flagH = 1;
		}
	}

	pos2.clear();
	pos2.push_back(Point(A2X, A2Y));
	pos2.push_back(Point(A2X + S1.widthSS, A2Y));
	pos2.push_back(Point(A2X, A2Y + S1.heightSS));
	pos2.push_back(Point(A2X + S1.widthSS, A2Y + S1.heightSS));

	S1.setIntersection(startx - pos1[0].x, endx - pos1[0].x, starty - pos1[0].y, endy - pos1[0].y);
	S2.setIntersection(startx - pos2[0].x, endx - pos2[0].x, starty - pos2[0].y, endy - pos2[0].y);

	widthCanvas		= maxX - minX;
	heightCanvas	= maxY - minY;
	S1.pointStart	= Point(pos1[0].x - minX, pos1[0].y - minY);
	S2.pointStart	= Point(pos2[0].x - minX, pos2[0].y - minY);
}

void Core::randomSIPosition(Share& S1, Share& S2){
	int x, y, spaceX, spaceY;

	spaceX	= (S1.intersection[1].x - S1.intersection[0].x) - widthSI;
	spaceY	= (S1.intersection[2].y - S1.intersection[0].y) - heightSI;

	srand((unsigned)time(0));
	x	= (rand() % (spaceX+1));
	y	= (rand() % (spaceY+1));

	S1.position.clear();
	S1.position.push_back(Point(S1.intersection[0].x + x, S1.intersection[0].y + y));
	S1.position.push_back(Point(S1.intersection[0].x + x + widthSI, S1.intersection[0].y + y));
	S1.position.push_back(Point(S1.intersection[0].x + x, S1.intersection[0].y + y + heightSI));
	S1.position.push_back(Point(S1.intersection[0].x + x + widthSI, S1.intersection[0].y + y + heightSI));

	S2.position.clear();
	S2.position.push_back(Point(S2.intersection[0].x + x, S2.intersection[0].y + y));
	S2.position.push_back(Point(S2.intersection[0].x + x + widthSI, S2.intersection[0].y + y));
	S2.position.push_back(Point(S2.intersection[0].x + x, S2.intersection[0].y + y + heightSI));
	S2.position.push_back(Point(S2.intersection[0].x + x + widthSI, S2.intersection[0].y + y + heightSI));
}

void Core::generateShareNIMSVCS(Share& S1, Share& S2){
	Mat C1			= generateNewImg(S1.heightSS*2, S1.widthSS*2, 1);
	Mat A1			= Mat::zeros(S1.heightSS*2, S1.widthSS*2, CV_8UC1);
	Mat C2			= generateNewImg(S2.rotHeightSS*2, S2.rotWidthSS*2, 1);
	Mat A2			= Mat::zeros(S2.rotHeightSS*2, S2.rotWidthSS*2, CV_8UC1);

	vector<Mat> channels1, channels2;
	unsigned seed;

	// outside intersection area for SS1
	// :::::::::::::::::::::::::::::::::::::::::
	cout << "\nGenerating area 1 of S1" << endl;
	// :::::::::::::::::::::::::::::::::::::::::
	for(int i = 0; i < S1.heightSS; i++){
		for(int j = 0; j < S1.widthSS; j++){
			int doit = 0;
			// assign matrix m except intersection area
			if((i >= S1.intersection[0].y) && (i < S1.intersection[2].y)){
				if((j < S1.intersection[0].x) || (j >= S1.intersection[1].x)){
					doit = 1;
				}
			} else {
				doit = 1;
			}
			if(doit){
				// obtain a time-based seed:
				// random permutation on 'position' to get matrix c
				seed = (chrono::system_clock::now().time_since_epoch().count())+j;
				shuffle (pos.begin(), pos.end(), default_random_engine(seed));

				int C[2][4];
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M0[u][v];
					}
				}

				// transform 1D to 2D matrix m
				int m[2][2];
				for(int l = 0; l < 4; l++){
					int rowm		= l/2;
					m[rowm][l % 2]	= C[0][pos[l]];
				}

				// asign matrix m to the share
				int marky = 0;
				for(int newy = i*2; newy < (i*2)+2; newy++){
					int markx = 0;
					for(int newx = j*2; newx < (j*2)+2; newx++){
						C1.at<uchar>(newy,newx) = m[marky][markx];
						if(m[marky][markx] == 255){
							A1.at<uchar>(newy,newx) = 0;
						} else {
							A1.at<uchar>(newy,newx) = 255;
						}
						markx = markx + 1;
					}
					marky = marky + 1;
				}
			}
		}
	}

	// outside intersection area for SS2
	// :::::::::::::::::::::::::::::::::::::::::
	cout << "\nGenerating area 1 of S2" << endl;
	// :::::::::::::::::::::::::::::::::::::::::
	for(int i = 0; i < S2.rotHeightSS; i++){
		for(int j = 0; j < S2.rotWidthSS; j++){
			int doit = 0;
			// assign matrix m except intersection area
			if((i >= S2.intersection[0].y) && (i < S2.intersection[2].y)){
				if((j < S2.intersection[0].x) || (j >= S2.intersection[1].x)){
					doit = 1;
				}
			} else {
				doit = 1;
			}
			if(doit){
				// obtain a time-based seed:
				// random permutation on 'position' to get matrix c
				seed = (chrono::system_clock::now().time_since_epoch().count())+j;
				shuffle (pos.begin(), pos.end(), default_random_engine(seed));

				int C[2][4];
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M0[u][v];
					}
				}

				// transform 1D to 2D matrix m
				int m[2][2];
				for(int l = 0; l < 4; l++){
					int rowm		= l/2;
					m[rowm][l % 2]	= C[1][pos[l]];
				}

				// asign matrix m to the share
				int marky = 0;
				for(int newy = i*2; newy < (i*2)+2; newy++){
					int markx = 0;
					for(int newx = j*2; newx < (j*2)+2; newx++){
						C2.at<uchar>(newy,newx) = m[marky][markx];
						if(m[marky][markx] == 255){
							A2.at<uchar>(newy,newx) = 0;
						} else {
							A2.at<uchar>(newy,newx) = 255;
						}
						markx = markx + 1;
					}
					marky = marky + 1;
				}
			}
		}
	}

	int siCoordinateY	= S1.position[0].y - S1.intersection[0].y;
	int siCoordinateX	= S1.position[0].x - S1.intersection[0].x;
	// inside intersection area but outside SI
	// :::::::::::::::::::::::::::::::::::::::::
	cout << "\nGenerating area 2" << endl;
	// :::::::::::::::::::::::::::::::::::::::::
	for(int i = 0; i < (S1.intersection[2].y - S1.intersection[0].y); i++){
		for(int j = 0; j < (S1.intersection[1].x - S1.intersection[0].x); j++){
			int doit			= 0;
			// assign matrix m except intersection area
			if((i >= siCoordinateY) && (i < (siCoordinateY + heightSI))){
				if((j < siCoordinateX) || (j >= siCoordinateX + widthSI)){
					doit = 1;
				}
			} else {
				doit = 1;
			}
			if(doit){
				// obtain a time-based seed:
				// random permutation on 'position' to get matrix c
				seed = (chrono::system_clock::now().time_since_epoch().count())+j;
				shuffle (pos.begin(), pos.end(), default_random_engine(seed));

				int C[2][4];
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M0[u][v];
					}
				}

				// transform 1D to 2D matrix m
				int m_1[2][2],m_2[2][2];
				for(int l = 0; l < 4; l++){
					int rowm		= l/2;
					m_1[rowm][l % 2]	= C[0][pos[l]];
					m_2[rowm][l % 2]	= C[1][pos[l]];
				}

				// asign matrix m to the share
				int marky = 0;
				for(int newy = i*2; newy < (i*2)+2; newy++){
					int markx = 0;
					for(int newx = j*2; newx < (j*2)+2; newx++){
						C1.at<uchar>(newy + (S1.intersection[0].y*2),newx + (S1.intersection[0].x*2)) = m_1[marky][markx];
						if(m_1[marky][markx] == 255){
							A1.at<uchar>(newy + (S1.intersection[0].y*2),newx + (S1.intersection[0].x*2)) = 0;
						} else {
							A1.at<uchar>(newy + (S1.intersection[0].y*2),newx + (S1.intersection[0].x*2)) = 255;
						}
						
						C2.at<uchar>(newy + (S2.intersection[0].y*2),newx + (S2.intersection[0].x*2)) = m_2[marky][markx];
						if(m_2[marky][markx] == 255){
							A2.at<uchar>(newy + (S2.intersection[0].y*2),newx + (S2.intersection[0].x*2)) = 0;
						} else {
							A2.at<uchar>(newy + (S2.intersection[0].y*2),newx + (S2.intersection[0].x*2)) = 255;
						}
						markx = markx + 1;
					}
					marky = marky + 1;
				}
			}
		}
	}

	// :::::::::::::::::::::::::::::::::::::::::
	cout << "\nGenerating area 3" << endl;
	// :::::::::::::::::::::::::::::::::::::::::
	for(int i = 0; i < heightSI; i++){
		for(int j = 0; j < widthSI; j++){
			// obtain a time-based seed:
			// random permutation on 'position' to get matrix c
			seed = (chrono::system_clock::now().time_since_epoch().count())+j;
			shuffle (pos.begin(), pos.end(), default_random_engine(seed));
			int C[2][4];
			// permasalahan ada di sini
			if(filteredSI.at<uchar>(i,j) < 128){
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M1[u][v];
					}
				}
			} else {
				for(int u = 0; u < 2; u++){
					for(int v = 0; v < 4; v++){
						C[u][v] = M0[u][v];
					}
				}
			}
			// transform 1D to 2D matrix m
			int m_1[2][2],m_2[2][2];
			for(int l = 0; l < 4; l++){
				int rowm		= l/2;
				m_1[rowm][l % 2]	= C[0][pos[l]];
				m_2[rowm][l % 2]	= C[1][pos[l]];
			}
			// asign matrix m to the share
			int marky = 0;
			for(int newy = i*2; newy < (i*2)+2; newy++){
				int markx = 0;
				for(int newx = j*2; newx < (j*2)+2; newx++){
					C1.at<uchar>(newy + ((S1.intersection[0].y + siCoordinateY)*2),newx + ((S1.intersection[0].x + siCoordinateX)*2)) = m_1[marky][markx];
					if(m_1[marky][markx] == 255){
						A1.at<uchar>(newy + ((S1.intersection[0].y + siCoordinateY)*2),newx + ((S1.intersection[0].x + siCoordinateX)*2)) = 0;
					} else {
						A1.at<uchar>(newy + ((S1.intersection[0].y + siCoordinateY)*2),newx + ((S1.intersection[0].x + siCoordinateX)*2)) = 255;
					}
					
					C2.at<uchar>(newy + ((S2.intersection[0].y + siCoordinateY)*2),newx + ((S2.intersection[0].x + siCoordinateX)*2)) = m_2[marky][markx];
					if(m_2[marky][markx] == 255){
						A2.at<uchar>(newy + ((S2.intersection[0].y + siCoordinateY)*2),newx + ((S2.intersection[0].x + siCoordinateX)*2)) = 0;
					} else {
						A2.at<uchar>(newy + ((S2.intersection[0].y + siCoordinateY)*2),newx + ((S2.intersection[0].x + siCoordinateX)*2)) = 255;
					}
					markx = markx + 1;
				}
				marky = marky + 1;
			}
		}
	}

	channels1.clear();
	channels1.push_back(C1);
    channels1.push_back(C1);
    channels1.push_back(C1);
	channels1.push_back(A1);

	channels2.clear();
	channels2.push_back(C2);
    channels2.push_back(C2);
    channels2.push_back(C2);
	channels2.push_back(A2);

	merge(channels1, S1.share);
	merge(channels2, S2.rotatedShare);
}

void Core::getNormalRotationImage(Share& S){
	if (S.revAngle == 0){
		S.share = S.rotatedShare.clone();
	} else {
		if (S.revAngle == 90){
			transpose(S.rotatedShare, S.share);  
			flip(S.share, S.share,1); //transpose+flip(1)=CW
		} else {
			if (S.revAngle == 270) {
				transpose(S.rotatedShare, S.share);  
				flip(S.share, S.share,0); //transpose+flip(0)=CCW
			} else {
				if (S.revAngle == 180){
					flip(S.rotatedShare, S.share,-1);    //flip(-1)=180
				}
			}
		}
	}
}

void Core::viewDecodingNIMSVCS(){
	setSharePath();
	string ssSize;
	for(int i = 0; i < sharesPath.size(); i++){
		vector<string> tempName = splitString(sharesPath[i], '/');
		ssSize		= tempName[tempName.size() - 1];
		// *************************************************************************************
		// ********************************* PRE-DECODING **************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "****************************** PRE-DECODING ***********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Input share 1 and share 2 " << endl;
		Share S1(1);
		S1.setShare(sharesPath[i]);
		cout << "\nS1's size = " << S1.width << "x" << S1.height << endl;

		Share S2(2);
		S2.setShare(sharesPath[i]);
		cout << "\nS2's size = " << S2.width << "x" << S2.height << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// =====================================================================================
		// ============================ ESTIMATING DECODING TIME ===============================
		// =====================================================================================
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "======================= ESTIMATING DECODING TIME ==============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Estimating total decoding time" << endl;
		long long int pos;

		pos = getPossibilities(S1,S2);
		cout << "\nPossibilities \t\t\t: " << pos << " positions" << endl;

		estimate1TimeRun(S1,S2);
		duration1run 	= duration1run + 1;
		printf("\nOne time run \t\t\t: %.2f seconds\n", duration1run);

		mpz_class poss(pos);
		mpz_class sec, min, hour, day, year;
		mpz_mul_ui(sec.get_mpz_t(),poss.get_mpz_t(),(duration1run*10)+0.5);
		mpz_div_ui(sec.get_mpz_t(),sec.get_mpz_t(),10);
		cout << "\nTotal time run (in second) \t: " << sec.get_ui() << " seconds" << endl;
		mpz_div_ui(min.get_mpz_t(),sec.get_mpz_t(),60);
		cout << "\nTotal time run (in minutes) \t: " << min.get_ui() << " minutes" << endl;
		mpz_div_ui(hour.get_mpz_t(),min.get_mpz_t(),60);
		cout << "\nTotal time run (in hours) \t: " << hour.get_ui() << " hours" << endl;
		mpz_div_ui(day.get_mpz_t(),hour.get_mpz_t(),24);
		cout << "\nTotal time run (in days) \t: " << day.get_ui() << " days" << endl;
		mpz_div_ui(year.get_mpz_t(),day.get_mpz_t(),365);
		cout << "\nTotal time run (in years) \t: " << year.get_ui() << " years" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// *************************************************************************************
		// ******************************** MANUAL STACKING ************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "***************************** MANUAL STACKING *********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// =====================================================================================
		// ==================================== STACKING =======================================
		// =====================================================================================
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "================================= STACKING ====================================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		manualStacking(S1,S2,sharesPath[i]);

		vector<string> tempPath = splitString(sharesPath[i],'/');
		string code				= tempPath[tempPath.size()-2];
		string folder			= tempPath[tempPath.size()-1];

		string tempPath1		= "result/decoding/nimsvcs/" + code + "/";
		_mkdir(tempPath1.c_str());
		string tempPath2		= tempPath1 + folder + "/";
		_mkdir(tempPath2.c_str());
		imwrite(tempPath2 + "ri.png",canvas);

		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================
		int position	= getIndexEstimation(S1, S2);
		mpz_class poss2(position);
		mpz_class sec2, min2, hour2, day2, year2;
		cout << endl;
		mpz_mul_ui(sec2.get_mpz_t(),poss2.get_mpz_t(),(duration1run*10)+0.5);
		mpz_div_ui(sec2.get_mpz_t(),sec2.get_mpz_t(),10);
		cout << "\Estimated time run (in second) \t\t: " << sec2.get_ui() << " seconds" << endl;
		mpz_div_ui(min2.get_mpz_t(),sec2.get_mpz_t(),60);
		cout << "\nEstimated time run (in minutes) \t: " << min2.get_ui() << " minutes" << endl;
		mpz_div_ui(hour2.get_mpz_t(),min2.get_mpz_t(),60);
		cout << "\nEstimated time run (in hours) \t\t: " << hour2.get_ui() << " hours" << endl;
		mpz_div_ui(day2.get_mpz_t(),hour2.get_mpz_t(),24);
		cout << "\nEstimated time run (in days) \t\t: " << day2.get_ui() << " days" << endl;
		mpz_div_ui(year2.get_mpz_t(),day2.get_mpz_t(),365);
		cout << "\nEstimated time run (in years) \t\t: " << year2.get_ui() << " years" << endl;
		cout << endl;

		fstream file;
		file.open(tempPath2 + "info.txt",fstream::out);
			file << "name : " << folder << endl;
			file << "duration 1 time run : " << duration1run << " seconds" << endl;
			file << "\nPossibilities : " << pos << " positions" << endl;
			file << "\nTotal duration (in second) : " << sec.get_ui() << " seconds" << endl;
			file << "\nTotal duration (in minute) : " << min.get_ui() << " minutes" << endl;
			file << "\nTotal duration (in hour) : " << hour.get_ui() << " hours" << endl;
			file << "\nTotal duration (in day) : " << day.get_ui() << " days" << endl;
			file << "\nTotal duration (in year) : " << year.get_ui() << " years" << endl;
			file << "\n\nIndex Position : " << position << endl;
			file << "\nEstimated time run (in second) : " << sec2.get_ui() << " seconds" << endl;
			file << "\nEstimated time run (in minute) : " << min2.get_ui() << " minutes" << endl;
			file << "\nEstimated time run (in hour) : " << hour2.get_ui() << " hours" << endl;
			file << "\nEstimated time run (in day) : " << day2.get_ui() << " days" << endl;
			file << "\nEstimated time run (in year) : " << year2.get_ui() << " years" << endl;
		file.close();
		S1.~Share();
		S2.~Share();
	}
}

void Core::viewDecodingNIMSVCS2(){
	setSharePath();
	setSecretImagePath();
	string ssSize;
	for(int i = 0; i < sharesPath.size(); i++){
		vector<string> tempName = splitString(sharesPath[i], '/');
		ssSize		= tempName[tempName.size() - 1];
		string type;
		if(siPath.size() == 1){
			singlePath	= siPath[0];
			type		= "1";
		} else {
			singlePath	= siPath[i];
			type		= "2";
		}
		setSecretImage(singlePath);
		// *************************************************************************************
		// ********************************* PRE-DECODING **************************************
		// *************************************************************************************
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n*******************************************************************************" << endl;
		cout << "****************************** PRE-DECODING ***********************************" << endl;
		cout << "*******************************************************************************" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n> Input share 1 and share 2 " << endl;
		Share S1(1);
		S1.setShare(sharesPath[i]);
		cout << "\nS1's size = " << S1.height << "x" << S1.width << endl;

		Share S2(2);
		S2.setShare(sharesPath[i]);
		cout << "\nS2's size = " << S2.height << "x" << S2.width << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// =====================================================================================

		// =====================================================================================
		// ============================ ESTIMATING DECODING TIME ===============================
		// =====================================================================================
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		cout << "\n===============================================================================" << endl;
		cout << "======================= ESTIMATING DECODING TIME ==============================" << endl;
		cout << "===============================================================================" << endl;
		// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
		// editing
		cout << "\n> Stacking based on brute force" << endl;
		vector<string> tempPath = splitString(sharesPath[i],'/');
		string code				= tempPath[tempPath.size()-2];
		string folder			= tempPath[tempPath.size()-1];

		string tempPath1		= "result/decoding/nimsvcs/" + code + "/";
		_mkdir(tempPath1.c_str());
		string tempPath2		= tempPath1 + folder + "/";
		_mkdir(tempPath2.c_str());
		double durationtot = 0;
		fstream file;
		file.open(tempPath2 + "durdecode.csv",fstream::out);
		for(int j = 0; j < (heightSI * 2); j++){
			cout << "Processing row : " << j << endl;
			for(int k = 0; k < (S2.width+S1.width - 1); k++){
				//stacking
				clock_t tStart1	= clock();
				S1.pointStart	= Point(S2.width - 1, S2.height - 1);
				S2.pointStart	= Point(k,j);
				getIntersectionSize(S1, S2);
				getNormalPointStart(S1, S2);
				estimateCanvas(S1, S2, 0);
				estimateStack(S1, S2, 0);
				double duration1	= (double)(clock() - tStart1)/CLOCKS_PER_SEC;
				durationtot			= durationtot + duration1;
				//resize
				if((widthIA >= (widthSI*2)) && (heightIA >= (heightSI*2))){
					Size size(200,200);//the dst image size,e.g.100x100
					Mat dst;//dst image
					resize(canvas,dst,size);//resize image
					canvas.release();
					
					file << j << "," << k << "," << fixed << setprecision(5) << durationtot << endl;
					stringstream ss;
					string name = tempPath2 + "stacked_";
					string type = ".jpg";
					ss<<name<<(j)<<"_"<<(k)<<type;

					string filename = ss.str();
					ss.str("");
					imwrite(filename,dst);

					dst.release();
				} else {
					canvas.release();
				}

				if(j == (heightSI * 2) - 1){
					if(k == (widthSI * 2) + 31){
						break;
					}
				}
			}
		}
		file.close();
		
		S1.~Share();
		S2.~Share();
	}
}

void Core::getIntersectionSize(Share& S1, Share& S2){
	if(S1.pointStart.x >= S2.pointStart.x){
		if((S1.pointStart.x + S1.width)<=S2.width){
			widthIA = S1.width;
		} else {
			widthIA = S2.pointStart.x + S2.width - S1.pointStart.x;
		}
	} else {
		if((S2.pointStart.x + S2.width)<=S1.width){
			widthIA = S2.width;
		} else {
			widthIA = S1.pointStart.x + S1.width - S2.pointStart.x;
		}
	}

	if(S1.pointStart.y >= S2.pointStart.y){
		if((S1.pointStart.y + S1.height)<=S2.height){
			heightIA = S2.height;
		} else {
			heightIA = S2.pointStart.y + S2.height - S1.pointStart.y;
		}
	} else {
		if((S2.pointStart.y + S2.height)<=S1.height){
			heightIA = S2.height;
		} else {
			heightIA = S1.pointStart.y + S1.height - S2.pointStart.y;
		}
	}
}
void Core::getNormalPointStart(Share& S1, Share& S2){
	if(S1.pointStart.x >= S2.pointStart.x){
		S1.pointStart.x = S1.pointStart.x - S2.pointStart.x;
		S2.pointStart.x = 0;
	} else {
		S2.pointStart.x = S2.pointStart.x - S1.pointStart.x;
		S1.pointStart.x = 0;
	}

	if(S1.pointStart.y >= S2.pointStart.y){
		S1.pointStart.y = S1.pointStart.y - S2.pointStart.y;
		S2.pointStart.y = 0;
	} else {
		S2.pointStart.y = S2.pointStart.y - S1.pointStart.y;
		S1.pointStart.y = 0;
	}
}

long long int Core::getPossibilities(Share S1, Share S2){
	long long int result;
	long int posX, posY;

	posX	= S1.width + S2.width - 1;
	posY	= S1.height + S2.height - 1;

	result	= (posX * posY) * 4;

	return result;
}

void Core::estimate1TimeRun(Share S1, Share S2){
	clock_t tStart;
	Point2i	pos1, pos2;
	int flag;

	tStart			= clock();
	S1.pointStart	= Point(S2.width - 1, S2.height - 1);
	S2.pointStart	= Point(0,0);
	estimateCanvas(S1, S2, 0);
	estimateStack(S1, S2, 0);
	//flag			= checkPattern();

	duration1run	= (double)(clock() - tStart)/CLOCKS_PER_SEC;
}

void Core::estimateCanvas(Share& S1, Share& S2, int r){
	int A1X, B1X, A2X, B2X, minX, maxX;
	A1X		= S1.pointStart.x;
	B1X		= S1.pointStart.x + S1.width;
	A2X		= S2.pointStart.x;
	if(r == 0){
		B2X		= S2.pointStart.x + S2.width;
	} else {
		if(r == 1){
			B2X		= S2.pointStart.x + S2.rotWidth;
		}
	}

	if((A1X >= A2X)&&(A1X < B2X)&&(B1X >= B2X)){
		minX	= A2X;
		maxX	= B1X;
	} else {
		if((A1X >= A2X)&&(A1X < B2X)&&(B1X < B2X)){
			minX	= A2X;
			maxX	= B2X;
		} else {
			if((A1X <= A2X)&&(B1X >= B2X)){
				minX	= A1X;
				maxX	= B1X;
			} else {
				if((A1X <= A2X)&&(B1X < B2X)&&(B1X > A2X)){
					minX	= A1X;
					maxX	= B2X;
				}
			}
		}
	}

	int A1Y, C1Y, A2Y, C2Y, minY, maxY;
	A1Y		= S1.pointStart.y;
	C1Y		= S1.pointStart.y + S1.height;
	A2Y		= S2.pointStart.y;
	if(r == 0){
		C2Y		= S2.pointStart.y + S2.height;
	} else {
		if(r == 1){
			C2Y		= S2.pointStart.y + S2.rotHeight;
		}
	}
	if((A1Y >= A2Y)&&(A1Y < C2Y)&&(C1Y >= C2Y)){
		minY	= A2Y;
		maxY	= C1Y;
	} else {
		if((A1Y >= A2Y)&&(A1Y < C2Y)&&(C1Y < C2Y)){
			minY	= A2Y;
			maxY	= C2Y;
		} else {
			if((A1Y <= A2Y)&&(C1Y >= C2Y)){
				minY	= A1Y;
				maxY	= C1Y;
			} else {
				if((A1Y <= A2Y)&&(C1Y < C2Y)&&(C1Y > A2Y)){
					minY	= A1Y;
					maxY	= C2Y;
				}
			}
		}
	}

	widthCanvas		= maxX - minX;
	heightCanvas	= maxY - minY;
}

void Core::estimateStack(Share& S1, Share& S2, int r){
	int positiony, positionx;
	vector<Mat> channelsBB;
	Mat C	= generateNewImg(heightCanvas, widthCanvas, 1);
	Mat A	= Mat::zeros(heightCanvas, widthCanvas, CV_8UC1);
	positiony = 0;
	// stacking share 1 to big box
	for(int i = 0; i < heightCanvas; i++){
		positionx = 0;
		for(int j = 0; j < widthCanvas; j++){
			if((i >= S1.pointStart.y) && (i < (S1.pointStart.y + S1.height))){
				if((j >= S1.pointStart.x)&&(j < (S1.pointStart.x + S1.width))){
					int currentP	= S1.channels[0].at<uchar>(positiony,positionx);
					if(currentP < 128){
						C.at<uchar>(i,j)	= S1.channels[0].at<uchar>(positiony,positionx);
						A.at<uchar>(i,j)	= 255;
					}
					positionx				= positionx + 1;
				}
			}
		}
		if((i >= S1.pointStart.y)&&(i < (S1.pointStart.y + S1.height))){
			positiony = positiony + 1;
		}
	}
	// stacking share 2 to big box
	positiony = 0;
	for(int i = 0; i < heightCanvas; i++){
		positionx = 0;
		for(int j = 0; j < widthCanvas; j++){
			if(r == 0){
				if((i >= S2.pointStart.y) && (i < (S2.pointStart.y + S2.height))){
					if((j >= S2.pointStart.x) && (j < (S2.pointStart.x + S2.width))){
						int currentP	= S2.channels[0].at<uchar>(positiony,positionx);
						if(currentP < 128){
							C.at<uchar>(i,j)	= S2.channels[0].at<uchar>(positiony,positionx);
							A.at<uchar>(i,j)	= 255;
						}
						positionx				= positionx + 1;
					}
				}
			} else {
				if(r == 1){
					if((i >= S2.pointStart.y) && (i < (S2.pointStart.y + S2.rotHeight))){
						if((j >= S2.pointStart.x) && (j < (S2.pointStart.x + S2.rotWidth))){
							int currentP	= S2.rotChannels[0].at<uchar>(positiony,positionx);
							if(currentP < 128){
								C.at<uchar>(i,j)	= S2.rotChannels[0].at<uchar>(positiony,positionx);
								A.at<uchar>(i,j)	= 255;
							}
							positionx				= positionx + 1;
						}
					}
				}
			}
		}
		if(r == 0){
			if((i >= S2.pointStart.y)&&(i < (S2.pointStart.y + S2.height))){
				positiony = positiony + 1;
			}
		} else {
			if(r == 1){
				if((i >= S2.pointStart.y)&&(i < (S2.pointStart.y + S2.rotHeight))){
					positiony = positiony + 1;
				}
			}
		}
	}
	channelsBB.push_back(C);
    channelsBB.push_back(C);
    channelsBB.push_back(C);
	channelsBB.push_back(A);

	merge(channelsBB, canvas);
}

void Core::manualStacking(Share& S1, Share& S2, string path){
	ifstream file;
	file.open(path + "info.txt");
	string str;
	vector<string> tempInfo;
	while (getline(file, str)){
		tempInfo.push_back(str);
	}
	
	vector<string> tempAngle = splitString(tempInfo[4], ':');
	stringstream(tempAngle[1]) >> S2.angle;

	cout << "\n> Rotating Normal 3OP of S2" << endl;

	getRotationImage(S2);
	S2.getRotationSize();

	vector<string> tempPos1 = splitString(tempInfo[tempInfo.size()-2], ':');
	tempPos1 = splitString(tempPos1[1], '(');
	tempPos1 = splitString(tempPos1[1], ')');
	tempPos1 = splitString(tempPos1[0], ',');
	int x1, y1;
	stringstream(tempPos1[0]) >> x1;
	stringstream(tempPos1[1]) >> y1;
	S1.pointStart	= Point(x1,y1);

	vector<string> tempPos2 = splitString(tempInfo[tempInfo.size()-1], ':');
	tempPos2 = splitString(tempPos2[1], '(');
	tempPos2 = splitString(tempPos2[1], ')');
	tempPos2 = splitString(tempPos2[0], ',');
	int x2, y2;
	stringstream(tempPos2[0]) >> x2;
	stringstream(tempPos2[1]) >> y2;
	S2.pointStart	= Point(x2,y2);

	estimateCanvas(S1, S2, 1);

	cout << "\n> Stacking The Shares" << endl;
	estimateStack(S1, S2, 1);
}

void Core::getRotationImage(Share& S){
	vector<Mat> tempChannels(4);
	
	if (S.angle == 0){
		S.rotatedShare = S.share.clone();
	} else {
		if (S.angle == 90){
			transpose(S.share, S.rotatedShare);  
			flip(S.rotatedShare, S.rotatedShare,1); //transpose+flip(1)=CW
		} else {
			if (S.angle == 270) {
				transpose(S.share, S.rotatedShare);  
				flip(S.rotatedShare, S.rotatedShare,0); //transpose+flip(0)=CCW
			} else {
				if (S.angle == 180){
					flip(S.share, S.rotatedShare,-1);    //flip(-1)=180
				}
			}
		}
	}

	split(S.rotatedShare,tempChannels);
	S.rotChannels	= tempChannels;
}

int Core::getIndexEstimation(Share& S1, Share& S2){
	int widthArea, heightArea, avWidthArea, avHeightArea, x1, y1, x2, y2;

	widthArea	= S2.rotWidth + S1.width - 1;
	heightArea	= S2.rotHeight + S1.height - 1;

	x1			= S2.rotWidth - 1;
	y1			= S2.rotHeight - 1;

	int alpha, beta;
	alpha		= x1 - S1.pointStart.x;
	beta		= y1 - S1.pointStart.y;

	x2			= S2.pointStart.x + alpha;
	y2			= S2.pointStart.y + beta;
	
	int index, position;
	index		= ((y2 - 1) * widthArea) + x2;
	position	= index + ((S2.angle/90) * (widthArea * heightArea));
	
	return position;
}