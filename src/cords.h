#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <omp.h>
using namespace std;

class choice
{
public:
	vector<int> id;
	string ch = "";
	int dvs = 0;
	vector<int> dv;
	float corr = 0;
	choice(vector<int> id)
	{
		sort(id.begin(), id.end());
		this->id = id;
	}
	choice(vector<int> id, int dvs, string choice, float corr, vector<int> dv)
	{
		sort(id.begin(), id.end());
		this->id = id;
		this->dvs = dvs;
		this->ch = choice;
		this->corr = corr;
		this->dv = dv;
	}
};

void CORDS(string ifnm, string ofnm, string scols);