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
    * \param PointList1 ��������1(ͼ������ϵ)
    * \param List1Num ��������1����
    * \param PointList2 ��������2(ͼ������ϵ)
    * \param List2Num ��������2����
    */
    BOOL CalcH(const std::vector<VectorFeaturePoint>& PointList0,const std::vector<VectorFeaturePoint>& PointList1,S32 MatchedNum);

    /**
    * \brief ͨ������������ڲ�������Ȳ���D�����˶�����(R,T,N,V)
    * \param CameraInterPara ����ڲ���
    * \param D ���(��λ:��)
    * \note �˼��㲽��������ɵ�Ӧ�Ծ������(���ѳɹ�����CalcHV())
    */
    BOOL CalcMotion(CameraInterCalc &CameraInterPara,F32 D);

    /**
    * \brief ʹ�õ�һ��ͼ���еĵ���������Ӧ�ĵ��ڵڶ���ͼ���ϵ�����
    * \param Point0 ��һ��ͼ���ϵĵ�����
    * \param Point1 ��Ӧ�ĵڶ���ͼ���ϵĵ�����
    * \return ����ɹ��򷵻�True
    * \note ���ô˺���ʱ�������CalcMotion()�ĵ���
    */
    BOOL CalcForwardPoint(const Point2D& Point0,Point2D& Point1);

    /**
    * \brief ʹ�õڶ���ͼ���еĵ���������Ӧ�ĵ��ڵ�һ��ͼ���ϵ�����
    * \param Point1 �ڶ���ͼ���ϵĵ�����
    * \param Point0 ��Ӧ�ĵ�һ��ͼ���ϵĵ�����
    * \return ����ɹ��򷵻�True
    * \note ���ô˺���ʱ�������CalcMotion()�ĵ���
    */
    BOOL CalcBackwardPoint(const Point2D& Point1,Point2D& Point0);

    /**
    * \brief ��Ӧ�Ծ���H[3*3]
    */
    Matrix H;
    Matrix Hinv;
    /**
    * \brief ��Ӧ������V[3]={up,vp,1}
    */
    Matrix V;
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
