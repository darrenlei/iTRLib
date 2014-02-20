#ifndef COMMFEATUREPOINT_H
#define COMMFEATUREPOINT_H

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{
/**
* brief ͨ���������ʾ��
*/
class CommFeaturePoint:public Point2D
{
public:
    /**
    * brief �����յ�������,��Ҫִ��Init��ſ�ʹ��.
    */
    CommFeaturePoint();
    /**
    * brief �ͷ�ռ�õ���Դ
    */
    virtual ~CommFeaturePoint();
    /**
    * brief �������(Ҫ������ά��һ��)
    */
    CommFeaturePoint(const CommFeaturePoint &other);
    /**
    * brief ��ȫ��Ĳ�����ʼ��������
    */
    void Init(Point2D Pos,S32 ID,F32 Dir,F32 Quality,F32 Value);
    /**
    * brief ����һ���������㷨�������㽨��(����,�ǵ��),�ɼ���������Ʒ���.
    */
    void Init(Point2D Pos,F32 Quality,F32 Value);
    /**
    * brief �����������ֻ��߷��������ı�ʾ
    */
    S32 ID;
    /**
    * brief ����ָ�򸽼�����
    */
    void *Tag;
    /**
    * brief ����(���ýǶ���,��+X������,ָ��+Y��Ϊ������)
    */
    F32 Dir;
    /**
    * brief ����,���ڱ�ʾ�����������Ч��,���㷨�ɲ��ò�ͬ���,һ����ֵԽ������Խ��.
    */
    F32 Quality;
    /**
    * brief ���㷨����һ��������ʽ���������ֵ�����ֵ����ʹ�ô���
    */
    F32 Value;
    /**
    * brief �Ƚ��������������Ч�ԡ����Ƚ�Quality
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
