#ifndef CAMERAINTERCALC_H
#define CAMERAINTERCALC_H

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{
    /**
    * \brief �ṩ����ڲ���������������������������任.
    * \note ��������ϵ����:ԭ����ͼ�����Ͻ�,X������,Y������.
    * \note �������ϵ����:ԭ���ڹ�������,X������,Y������.
    */
    class CameraInterCalc
    {
        public:
            /**
            * \brief ��ʼ����׼������ڲ�������
            */
            CameraInterCalc();
            /**
            * \brief �ͷ�
            */
            virtual ~CameraInterCalc();
            /**
            * \brief ����Clone���캯��
            */
            CameraInterCalc(const CameraInterCalc &other);
            /**
            * \brief ͨ�����ý���ķ������ɾ���
            * \param F ����(��λ:��)
            * \param dX ˮƽ��������λ�ֱ���(��λ:��/����)
            * \param dY ��ֱ��������λ�ֱ���(��λ:��/����)
            * \param u0 ˮƽ���ͼ���������(��λ:����)
            * \param v0 ��ֱ���ͼ���������(��λ:����)
            */
            void SetPara(F32 F,F32 dX,F32 dY,F32 u0,F32 v0);
            /**
            * \brief ͨ�������ӳ��нǵķ������ɾ���
            * \param degX ˮƽ�����ӳ��н�(��λ:�Ƕ�)
            * \param degY ��ֱ�������ӳ��н�(��λ:�Ƕ�)
            * \param u0 ˮƽ���ͼ���������(��λ:����)
            * \param v0 ��ֱ���ͼ���������(��λ:����)
            */
            void SetPara(F32 degX,F32 degY,F32 u0,F32 v0);
            /**
            * \brief ������굽���������ת��
            * \param CameraPoint ������������ϵ����([x,y,z],��λ:��)
            * \param PixelPoint �������������ϵ([u,v,1],��λ:����)
            */
            BOOL CalcC2P(const Vector &CameraPoint,Vector &PixelPoint);
            /**
            * \brief �������굽��������ת��
            * \param PixelPoint �������������ϵ([u,v,1],��λ:����)
            * \param Z Ϊ��������µļٶ�����
            * \param CameraPoint ������������ϵ����([x,y,z],��λ:��)
            */
            BOOL CalcP2C(const Vector &PixelPoint,F32 Z,Vector &CameraPoint);
            /**
            * \note ������굽���������ת������
            */
            Matrix MatC2P;
            /**
            * \note �������굽��������ת������
            */
            Matrix MatP2C;
        protected:
        private:
            /**
            * \note �ڵ���SetPara����Ч
            */
            BOOL isMatC2PAvailable;
            /**
            * \note �ڵ���CalcP2Cʱ�Զ�ת����ʹ����Ч(����isMatC2PAvailable��Чʱ����Ч)
            */
            BOOL isMatP2CAvailable;
    };
}

#endif // CAMERAINTERCALC_H
