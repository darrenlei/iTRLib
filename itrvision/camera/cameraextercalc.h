#ifndef CAMERAEXTERCALC_H
#define CAMERAEXTERCALC_H

#include "itrbase.h"
#include "../feature/feature.h"
#include "camera.h"

using namespace itr_math;

namespace itr_vision
{
/*
    �������������������
*/
class CameraExterCalc
{
    public:
        /*
            �������ݳ�ʼ��
        */
        CameraExterCalc();
        /*
            ������Դ
        */
        virtual ~CameraExterCalc();
        /*
            ���ݸ���
        */
        CameraExterCalc(const CameraExterCalc& other);
        /*
            ʹ������������ͨ��RANSAC���㵥Ӧ�Ծ���(H,V)
        */
        BOOL CalcHV(CommFeaturePoint* PointList1,S32 List1Num,CommFeaturePoint* PointList2,S32 List2Num);
        /*
            ͨ������������ڲ�������Ȳ���D�����˶�����(R,T,N)
        */
        BOOL CalcMotion(CameraInterCalc& CameraInterPara,F32 D);
        Matrix H;
        Vector V;
        Vector R;
        Vector T;
        Vector N;
    protected:
    private:
};
}
#endif // CAMERAEXTERCALC_H
