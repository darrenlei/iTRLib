#ifndef VECTORFEATUREPOINT_H
#define VECTORFEATUREPOINT_H

#include "itrbase.h"
#include "commfeaturepoint.h"

using namespace itr_math;

namespace itr_vision
{
/**
* breif ������SIFT&SURF���㷨������������
*/
class VectorFeaturePoint:public CommFeaturePoint
{
public:
    /**
    * brief ��ʼ���յĶ���
    */
    VectorFeaturePoint();
    /**
    * brief �ͷ�ռ�õ���Դ
    */
    virtual ~VectorFeaturePoint();
    /**
    * brief ���ƶ���(Ҫ������ά��һ��)
    * param other ���ƽ��
    */
    VectorFeaturePoint(const VectorFeaturePoint &other);
    /**
    * brief ��ȫ��Ĳ�����ʼ��������
    */
    void Init(Point2D Pos,S32 ID,F32 Dir,F32 Quality,F32 Value,S32 FeatureDim,S32 LevelNo,F32 Scale);
    /**
    * brief �Գ��ò�����ʼ��������
    */
    void Init(Point2D Pos,F32 Dir,S32 FeatureDim,S32 LevelNo,F32 Scale);
    /**
    * brief ������ʽ���������ֵ�����ֵ����ʹ�ô���
    */
    Vector Feature;
    /**
    * brief ���ڲ���
    */
    S32 LevelNo;
    /**
    * brief �߶�ϵ��
    */
    F32 Scale;
protected:
private:
};

}

#endif // VECTORFEATUREPOINT_H
