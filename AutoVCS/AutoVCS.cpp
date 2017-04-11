#include "stdafx.h"
#include "targetver.h"
#include "Core.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>     // std::cout
#include <vector>       // std::vector
#include <string>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace std;

int main()
{
	srand((unsigned)time(0));
	Core c;
	int opt = c.viewMainMenu();

	if(opt == 1){
		c.viewEncoding();
	} else {
		if(opt == 2){
			c.viewDecoding();
		} else {
			exit(0);
		}
	}

	cout << "\n";
	system("pause");
	return 0;
}