#ifndef CAMERAEXTERCALC_H
#define CAMERAEXTERCALC_H

#include "itrbase.h"
#include "../feature/feature.h"
#include "camera.h"
#include "itralgorithm.h"
#include "math.h"

using namespace itr_math;
using namespace itr_algorithm;
namespace itr_vision
{
/**
* \brief �������������������
* \note ʵ��ͨ��������ƥ����㵥Ӧ�Ծ���(H)
* \note ʵ��ͨ������������Ϣ�����˶�����(R,T,N,V)
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
    * \brief ʹ������������ͨ��RANSAC���㵥Ӧ�Ծ���(H)
    * \param PointList1 ��������1
    * \param List1Num ��������1����
    * \param PointList2 ��������2
    * \param List2Num ��������2����
    */
    BOOL CalcH(VectorFeaturePoint *PointList1,S32 List1Num,VectorFeaturePoint *PointList2,S32 List2Num,S32 matched_num);

    /**
    * \brief ͨ������������ڲ�������Ȳ���D�����˶�����(R,T,N,V)
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
    * \note ת����������3*3*2��
    */
    Matrix R;
    /**
    * \note ƽ������������3*2��
    */
    Matrix t;
    /**
    * \note ƽ��ĵ�λ������������3*2��
    */
    Matrix N;
protected:
private:
};
}
#endif // CAMERAEXTERCALC_H
