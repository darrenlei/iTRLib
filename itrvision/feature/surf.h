#ifndef SURF_H
#define SURF_H

#include <vector>
#include "itrbase.h"
#include "./feature.h"
#include "../process/process.h"

using namespace itr_math;

namespace itr_vision
{
static const S32 SURF_OCTAVES = 5;
static const S32 SURF_INTERVALS = 4;
static const S32 SURF_INIT_SAMPLE = 2;
static const F32 SURF_THRESHOLD = 0.0004f;
static const S32 SURF_Filter_Map [SURF_OCTAVES][SURF_INTERVALS] = {{0,1,2,3}, {1,3,4,5}, {3,5,6,7}, {5,7,8,9}, {7,9,10,11}};
static const F64 SURF_Gauss25 [7][7] =
{
    {0.02546481,	0.02350698,	0.01849125,	0.01239505,	0.00708017,	0.00344629,	0.00142946},
    {0.02350698,	0.02169968,	0.01706957,	0.01144208,	0.00653582,	0.00318132,	0.00131956},
    {0.01849125,	0.01706957,	0.01342740,	0.00900066,	0.00514126,	0.00250252,	0.00103800},
    {0.01239505,	0.01144208,	0.00900066,	0.00603332,	0.00344629,	0.00167749,	0.00069579},
    {0.00708017,	0.00653582,	0.00514126,	0.00344629,	0.00196855,	0.00095820,	0.00039744},
    {0.00344629,	0.00318132,	0.00250252,	0.00167749,	0.00095820,	0.00046640,	0.00019346},
    {0.00142946,	0.00131956,	0.00103800,	0.00069579,	0.00039744,	0.00019346,	0.00008024}
};
/**
* \brief ����ʵ����SURF�㷨
* \note ͼ�����ݻ���itr_math::Matrix����,ȫ��Ϊ�����ȸ������.
* \note ������̾�������itr_math::Calculate�еķ���,�Ӷ�ʵ���˼��㷽ʽ�Ŀ�����Ż�.
* \note ���ڻῼ�ǽ��ж�ARM-Neon�Ļ���Ż�.
*/
class SURF
{
public:
    /**
    * \brief ��ɶ������������Init()������������.
    */
    SURF();
    /**
    * \brief �ͷŶ�̬������ڴ�
    */
    virtual ~SURF();
    /**
    * \brief ��ʼ��SURF��,�ڲ�����,ֻ�гɹ����ô˺����������������.
    * \param Width ͼ����
    * \param Height ͼ��߶�
    * \param OctaveNum �˶���
    * \param IntervalNum �����
    * \param InitSample ��ʼ����
    * \param Threshold ���˷�ֵ
    */
    void Init(S32 Width,S32 Height,S32 OctaveNum,S32 IntervalNum,S32 InitSample,F32 Threshold);
    /**
    * \brief ����ͼ������������
    * \param Img ������ͼ��
    * \param FeaturePointList �������б�(���ɵ�������ᴢ������)
    * \return �ҵ�������������
    */
    S32 Process(const Matrix& Img,std::vector<VectorFeaturePoint>& FeaturePointList);
    /**
    * \brief ���˷�ֵ
    */
    F32 Threshold;

    std::vector<BoxHessian*> OctaveList;
private:
    BOOL IsExtremum(S32 r, S32 c, BoxHessian *t, BoxHessian *m, BoxHessian *b);
    void MakeFeaturePoint(S32 r, S32 c, BoxHessian *t, BoxHessian *m, BoxHessian *b,VectorFeaturePoint &vfp);
    void GetOrientation(VectorFeaturePoint& Point);
    void GetDescriptor(VectorFeaturePoint& Point);
    F32 HaarX(S32 row, S32 column, S32 s);
    F32 HaarY(S32 row, S32 column, S32 s);
    //std::vector<BoxHessian*> OctaveList;
    Matrix IntImg;
    S32 OctaveNum;
    S32 IntervalNum;
    S32 InitSample;
};
}
#endif // SURF_H
