// adaboost_train.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "train.h"
#include "MemTracker.h"

int main()
{
	vector<TRAINDATA> trainlist;
	CvSize trainSize;
	MemTracker tracker;

	trainSize.height = 24;
	trainSize.width = 24;
	DATALIMIT datalimit;
	datalimit.lnegcnt = 4;
	datalimit.lposcnt = 4;
	
	bool verbose = true;
	CvMat** gabortab = create_gabor(trainSize, verbose);
	MAPPING map = getmapping(8, GETMAPPING_U2);
	
	vector<FILENAME> ftab1 = rdFNameFromDir(TRAIN_POS_PATH, "*.bmp");
	vector<FILENAME> ftab2 = rdFNameFromDir(TRAIN_NEG_PATH, "*.bmp");
	
	int poscnt = ftab1.size();
	int negcnt = ftab2.size();
	for (int i=0; i<negcnt; i++){
		
		ftab1.push_back(ftab2[i]);
	}

	int allcnt = poscnt + negcnt;
	float *featab;

	featab = (float *)malloc( sizeof(float)*allcnt*FEATURE_GABOR_LBP );
	memset(featab, 0, ( sizeof(float)*allcnt*FEATURE_GABOR_LBP ) );

	trainlist = vj_collect_data(ftab1, poscnt, gabortab, map, trainSize, datalimit, featab);
	
	CLASSIFIER.polarity =               zeros(1,T);
	CLASSIFIER.theta =                  zeros(1,T);
	CLASSIFIER.alpha =                  zeros(1,T);
	CLASSIFIER.w =                      [];
	CLASSIFIER.fast =                   [];
	CLASSIFIER.IMSIZE =                 WEAK.IMSIZE;
	CLASSIFIER.type =                   'SINGLE';  % specify if this is a cascade or single classifier
	CLASSIFIER.WLDA     =               cell(1,T);


	free(featab);
	featab = NULL;
	
	return 0;
}