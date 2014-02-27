#ifndef CAMERASTEREOCALC_H
#define CAMERASTEREOCALC_H

#include "itrbase.h"
#include "itralgorithm.h"
#include "../feature/feature.h"
#include "./cameraintercalc.h"

using namespace itr_math;

namespace itr_vision
{
/**
* \brief �������ڽ��������Ӿ�����,���Եó������������ԵĹ�ͬƽ��������ľ���.
* \note ���㷨�������ͬ��װ��һ������֪�ĸ�����,��ͼ����������Ʒ��ҪΪһƽ��,��ɹ����ƽ��ķ���.
*/
class CameraStereoCalc
{
public:
    /**
    * \brief ��ʾ����ó��ĸ�����Ϣ
    */
    struct CalcExInfo
    {
        /**
        * \brief ������ƥ���
        * \note ƥ������*2/����������*100%
        */
        F32 MatchPercent;
        /**
        * \brief ƽ�淽�̵ķ���
        */
        F32 Var;
        /**
        * \brief ƽ�淽������
        * \note Ax+By+Cz=1;Equ[0~2]=A,B,C.
        */
        Vector Equ;
    };
    CameraStereoCalc();
    virtual ~CameraStereoCalc();
    /**
    * \brief ��ʼ�����㷨������Ĳ���
    * \param CameraInterCalc0 ���0���ڲ���
    * \param CameraInterCalc1 ���1���ڲ���
    * \param Distance ���ԭ����(��λ:��)
    * \note **���ڴ˲���ʹ�õ�����ϵ��ϵ**
    */
    void Init(CameraInterCalc* CameraInterCalc0,CameraInterCalc* CameraInterCalc1,F32 Distance);
    /**
    * \brief ʹ������������ͨ��RANSAC����ƽ�淽��
    * \param PointList1 ��������1(ͼ������ϵ)
    * \param List1Num ��������1����
    * \param PointList2 ��������2(ͼ������ϵ)
    * \param List2Num ��������2����
    * \param DeepZero ����X=0,Y=0��Z��ֵ
    * \return �Ƿ�ɹ�����
    */
    bool Calc(VectorFeaturePoint *PointList1,S32 List1Num,VectorFeaturePoint *PointList2,S32 List2Num,F32* DeepZero);
    /**
    * \brief �ڳɹ�ִ��Calc���ͨ���˺�����ø��ӵļ�������Ϣ
    */
    CalcExInfo GetCalcExInfo();
    /**
    * \brief ���0���ڲ���
    */
    CameraInterCalc* CameraInterCalc0;
    /**
    * \brief ���1���ڲ���
    */
    CameraInterCalc* CameraInterCalc1;
    /**
    * \brief ���ԭ����(��λ:��)
    */
    F32 Distance;
    /**
    * \brief ���Z�ᰲװ�Ƕ�(���������ʹ�ù�ͬ�İ�װ�Ƕ�)
    * \note ��λ:�Ƕ�,ʹ����������ϵ.
    */
    F32 InstallAngle;
protected:
private:
};

}

#endif // CAMERASTEREO_H
