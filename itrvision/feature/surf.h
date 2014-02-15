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

    /**
    * /brief ����ʵ����SURF�㷨
    * /note ͼ�����ݻ���itr_math::Matrix����,ȫ��Ϊ�����ȸ������.
    * /note ������̾�������itr_math::Calculate�еķ���,�Ӷ�ʵ���˼��㷽ʽ�Ŀ�����Ż�.
    * /note ���ڻῼ�ǽ��ж�ARM-Neon�Ļ���Ż�.
    */
    class SURF
    {
        public:
            /**
            * /brief ��ɶ������������Init()������������.
            */
            SURF();
            /**
            * /brief �ͷŶ�̬������ڴ�
            */
            virtual ~SURF();
            /**
            * /brief ��ʼ��SURF��,�ڲ�����,ֻ�гɹ����ô˺����������������.
            * /param Width ͼ����
            * /param Height ͼ��߶�
            * /param OctaveNum �˶���
            * /param IntervalNum �����
            * /param InitSample ��ʼ����
            * /param Threshold ���˷�ֵ
            */
            void Init(S32 Width,S32 Height,S32 OctaveNum,S32 IntervalNum,S32 InitSample,F32 Threshold);
            /**
            * /brief ����ͼ������������
            * /param Img ������ͼ��
            * /param FeaturePointList �������б�(���ɵ�������ᴢ������)
            * /return �ҵ�������������
            */
            S32 Process(const Matrix& Img,std::vector<VectorFeaturePoint> FeaturePointList);
            /**
            * /brief ���˷�ֵ
            */
            F32 Threshold;
        private:
            BOOL IsExtremum(S32 r, S32 c, BoxHessian *t, BoxHessian *m, BoxHessian *b);
            void MakeFeaturePoint(S32 r, S32 c, BoxHessian *t, BoxHessian *m, BoxHessian *b);
            std::vector<BoxHessian*> OctaveList;
            Matrix IntImg;
            S32 OctaveNum;
            S32 IntervalNum;
            S32 InitSample;
    };
}
#endif // SURF_H
