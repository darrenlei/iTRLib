
#define TRAIN_POS_NUM 3000     // number of positive examples in the training set
#define TRAIN_NEG_NUM 5000     // number of negative examples in the training set
#define TEST_POS_NUM  3000
#define TEST_NEG_NUM  5000
#define MAX_IMAGENAME_LEN 64
#define MAX_FULLPATHNAME_LEN 256

#define TRAIN_POS_PATH "D:\\rmb\\¸ºÑù±¾\\"
#define TRAIN_NEG_PATH "Data\\face_databases\\combined\\train\\non-face\\"
#define TEST_POS_PATH  "Data\\face_databases\\combined\\test\\face\\"
#define TEST_NEG_PATH  "Data\\face_databases\\combined\\test\\non-face\\"

#define GABOR_PIC_COUNT 40
#define LBP_CODE_LEN 59
#define FEATURE_GABOR_LBP (GABOR_PIC_COUNT*LBP_CODE_LEN)

#define	CLASS_POSITIVE 1
#define CLASS_NEGETIVE 2

#define PI 3.1415926
#define MININF_DB -999999
#define MAXINF_DB 999999

//#define DEBUG_PRINTF

#define MAX_TABLE_LEN 256
#define INREGION(value, flimit, climit) ( (value)>=(flimit) && (value)<=(climit) )


/********************************************/
// used for func "getmapping" as input parameters.
#define GETMAPPING_U2   0
#define GETMAPPING_RI   1
#define GETMAPPING_RIU2 2

/********************************************/

/********************************************/
// used for func "lbp" as input parameters.
#define LBP_HIST          0
#define LBP_NORMALIZE     1
#define LBP_NORMALIZEHIST 2

/********************************************/

/********************************************/

#define WEAK_CLASSIFIER_LENGTH 40


/********************************************/

typedef struct _filename
{
	char fn[MAX_IMAGENAME_LEN];
}FILENAME;

typedef struct datalimit{
	
	long lposcnt;
	long lnegcnt;
}DATALIMIT;

typedef struct elem{

	float v;
	int index;
}ELEM;

typedef struct feat{

	int dim;
	union
	{
		uchar* ptr;
		short* s;
		int* i;
		float* fl;
		double* db;
	} data;

}FEAT;

typedef FEAT VECTOR;

typedef struct train{

	CvMat *pic;        // train img matric 
	FEAT fea;          // train img feature data;
	int clas;          // train img class.
}TRAINDATA;

typedef struct mapping{

	int table[MAX_TABLE_LEN];
	int sample;
	int num;
}MAPPING;

typedef struct wlda{
	
	float w[WEAK_CLASSIFIER_LENGTH*LBP_CODE_LEN];
}WLDA;

typedef struct wandprojres
{
	VECTOR w;		//wlda vector
	VECTOR projres;	//project result

}WANDPROJRES;

typedef struct weak{

	float    theta[WEAK_CLASSIFIER_LENGTH];
	float   minerr[WEAK_CLASSIFIER_LENGTH];
	float polarity[WEAK_CLASSIFIER_LENGTH];
	CvSize imsize;
	WLDA wlda;
}WEAK;

typedef struct classifier{

	float *polarity;
	float *theta;
	float *alpha;
	float *w;
	float *fast;
	CvSize imsize;
	int type;
	WLDA wlda;

}CLASSIFIER;