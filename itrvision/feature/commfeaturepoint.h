#ifndef COMMFEATUREPOINT_H
#define COMMFEATUREPOINT_H

#include "itrbase.h"
using namespace itr_math;

namespace itr_vision
{
/*
    ͨ���������ʾ��
*/
class CommFeaturePoint
{
    public:
        /*
            �����յ�������,��Ҫִ��Init��ſ�ʹ��.
        */
        CommFeaturePoint();
        /*
            �ͷ�ռ�õ���Դ
        */
        virtual ~CommFeaturePoint();
        /*
            �������(Ҫ������ά��һ��)
        */
        CommFeaturePoint(const CommFeaturePoint& other);
        /*
            ��ȫ��Ĳ�����ʼ��������
        */
        void Init(S32 ID,Point2D Pos,F32 Dir,F32 Quality,F32 Value,S32 FeatureDim);
        /*
            ����һ���������㷨�������㽨��(����,�ǵ��),�ɼ���������Ʒ���.
        */
        void Init(Point2D Pos,F32 Quality);
        /*
            ����������������ʾ���㷨�������㽨��(SIFT��SURF��)
        */
        void Init(Point2D Pos,F32 Dir,F32 Quality,S32 FeatureDim);
        /*
            �����������ֻ��߷��������ı�ʾ
        */
        S32 ID;
        /*
            ������λ��(������ӦGeometry���к�������)
        */
        Point2D Pos;
        /*
            ����(���ýǶ���,��+X������,ָ��+Y��Ϊ������)
        */
        F32 Dir;
        /*
            ����,���ڱ�ʾ�����������Ч��,���㷨�ɲ��ò�ͬ���,һ����ֵԽ������Խ��.
        */
        F32 Quality;
        /*
            ���㷨����һ��������ʽ���������ֵ�����ֵ����ʹ�ô���
        */
        F32 Value;
        /*
            ����㷨����һ��������ʽ���������ֵ�����ֵ����ʹ�ô���
        */
        Vector Feature;
        /*
            ����ָ�򸽼�����
        */
        void* Tag;
    protected:
    private:
};
}
#endif // COMMFEATUREPOINT_H
