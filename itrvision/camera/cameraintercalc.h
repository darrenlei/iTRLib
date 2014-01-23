#ifndef CAMERAINTERCALC_H
#define CAMERAINTERCALC_H

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{
/*
    �ṩ����ڲ���������������������������任
*/
class CameraInterCalc
{
    public:
        /*
            ��ʼ���ڲ����ݿռ�(��������)
        */
        CameraInterCalc();
        /*
            �ͷ���Դ
        */
        virtual ~CameraInterCalc();
        /*
            �������ݸ���
        */
        CameraInterCalc(const CameraInterCalc& other);
        /*
            ͨ�����ý���ķ������ɾ���
        */
        void SetPara(F32 F,F32 dX,F32 dY,F32 u0,F32 v0);
        /*
            ͨ�����ù�һ������ķ�ʽ���ɾ���
        */
        void SetPara(F32 Fu,F32 Fv,F32 u0,F32 v0);
        /*
            ������굽���������ת��
        */
        BOOL CalcC2P(const Vector& CameraPoint,Vector& PixelPoint);
        /*
            �������굽��������ת��(ZΪ��������µļٶ�����)
        */
        BOOL CalcP2C(const Vector& PixelPoint,F32 Z,Vector& CameraPoint);
        /*
            ������굽���������ת������
        */
        Matrix MatC2P;
        /*
            �������굽��������ת������
        */
        Matrix MatP2C;
    protected:
    private:
        BOOL isMatC2PAvailable;//�ڵ���SetPara��Ч
        BOOL isMatP2CAvailable;//�ڵ���CalcP2Cʱ�Զ�ת����ʹ����Ч(����isMatC2PAvailable��Чʱ����Ч)
};
}

#endif // CAMERAINTERCALC_H
