#ifndef VECTORFEATUREPOINT_H
#define VECTORFEATUREPOINT_H

#include "itrbase.h"
#include "commfeaturepoint.h"

using namespace itr_math;

namespace itr_vision
{
    /*
        ������SIFT&SURF���㷨������������
    */
    class VectorFeaturePoint:public CommFeaturePoint
    {
        public:
            /*
                ��ʼ���յĶ���
            */
            VectorFeaturePoint();
            /*
                �ͷ�ռ�õ���Դ
            */
            virtual ~VectorFeaturePoint();
            /*
                ���ƶ���(Ҫ������ά��һ��)
            */
            VectorFeaturePoint(const VectorFeaturePoint &other);
            /*
                ��ȫ��Ĳ�����ʼ��������
            */
            void Init(Point2D Pos,S32 ID,F32 Dir,F32 Quality,F32 Value,S32 FeatureDim,S32 LevelNo,F32 Scale);
            /*
                �Գ��ò�����ʼ��������
            */
            void Init(Point2D Pos,F32 Dir,S32 FeatureDim,S32 LevelNo,F32 Scale);
            /*
                ������ʽ���������ֵ�����ֵ����ʹ�ô���
            */
            Vector Feature;
            /*
                ���ڲ���
            */
            S32 LevelNo;
            /*
                �߶�ϵ��
            */
            F32 Scale;
        protected:
        private:
    };

}

#endif // VECTORFEATUREPOINT_H
