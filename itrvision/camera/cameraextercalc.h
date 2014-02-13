#ifndef CAMERAEXTERCALC_H
#define CAMERAEXTERCALC_H

#include "itrbase.h"
#include "../feature/feature.h"
#include "camera.h"

using namespace itr_math;

namespace itr_vision
{
    /**
    * \brief �������������������
    * \note ʵ��ͨ��������ƥ����㵥Ӧ�Ծ���(H,V)
    * \note ʵ��ͨ������������Ϣ�����˶�����(R,T,N)
    */
    class CameraExterCalc
    {
        public:
            /**
            * \brief �ڲ��������ݳ�ʼ��
            */
            CameraExterCalc();
            /**
            * \brief ������Դ
            */
            virtual ~CameraExterCalc();
            /**
            * \brief ����Clone���캯��
            */
            CameraExterCalc(const CameraExterCalc &other);
            /**
            * \brief ʹ������������ͨ��RANSAC���㵥Ӧ�Ծ���(H,V)
            * \param PointList1 ��������1
            * \param List1Num ��������1����
            * \param PointList2 ��������2
            * \param List2Num ��������2����
            */
            BOOL CalcHV(VectorFeaturePoint *PointList1,S32 List1Num,VectorFeaturePoint *PointList2,S32 List2Num);
            /**
            * \brief ͨ������������ڲ�������Ȳ���D�����˶�����(R,T,N)
            * \param CameraInterPara ����ڲ���
            * \param D ���(��λ:��)
            * \note �˼��㲽��������ɵ�Ӧ�Ծ������(���ѳɹ�����CalcHV())
            */
            BOOL CalcMotion(CameraInterCalc &CameraInterPara,F32 D);
            /**
            * \brief ��Ӧ�Ծ���H[3*3]
            */
            Matrix H;
            /**
            * \brief ��Ӧ������V[3]={up,vp,1}
            */
            Vector V;
            /**
            * \note ��ο�����������55ҳ,����ɼ��㲢�޸Ĵ�ע��
            */
            Vector R;
            /**
            * \note ��ο�����������55ҳ,����ɼ��㲢�޸Ĵ�ע��
            */
            Vector T;
            /**
            * \note ��ο�����������55ҳ,����ɼ��㲢�޸Ĵ�ע��
            */
            Vector N;
        protected:
        private:
    };
}
#endif // CAMERAEXTERCALC_H
