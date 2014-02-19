#ifndef COMMFEATUREPOINT_H
#define COMMFEATUREPOINT_H

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{
/*
    ͨ���������ʾ��
*/
class CommFeaturePoint:public Point2D
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
    CommFeaturePoint(const CommFeaturePoint &other);
    /*
        ��ȫ��Ĳ�����ʼ��������
    */
    void Init(Point2D Pos,S32 ID,F32 Dir,F32 Quality,F32 Value);
    /*
        ����һ���������㷨�������㽨��(����,�ǵ��),�ɼ���������Ʒ���.
    */
    void Init(Point2D Pos,F32 Quality,F32 Value);

    /*
        �����������ֻ��߷��������ı�ʾ
    */
    S32 ID;
    /*
        ����ָ�򸽼�����
    */
    void *Tag;
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
        �Ƚ��������������Ч�ԡ����Ƚ�Quality
    */
    bool operator>(const CommFeaturePoint &other) const
    {
        return this->Quality>other.Quality;
    }
    bool operator<(const CommFeaturePoint &other) const
    {
        return this->Quality<other.Quality;
    }
protected:
private:
};
}
#endif // COMMFEATUREPOINT_H
