#ifndef BOXHESSIAN_H
#define BOXHESSIAN_H

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{
/**
* \brief ����ʹ�ú�ʽ�����˲����ķ��������������ͼ���Hessianֵ
*/
class BoxHessian
{
public:
    /**
    * \brief ��ʼ����������,���������Init()�ſ�����ʹ��.
    */
    BoxHessian();
    /**
    * \brief ������Դ
    */
    virtual ~BoxHessian();
    /**
    * \brief ����Hessianֵ����.
    */
    F32* GetHessianData() const;
    /**
    * \brief ����Laplacianֵ����.
    */
    U8* GetLaplacianData() const;
    /**
    * \brief ���س�ʼ�Ŀ��
    */
    S32 GetWidth() const;
    /**
    * \brief ���س�ʼ�ĸ߶�
    */
    S32 GetHeight() const;
    /**
    * \brief ��ʼ��BoxHessian�˲���
    * \param Width �����ͼ����
    * \param Height �����ͼ��߶�
    * \param Step �˲�������ֵ
    * \param FilterLength �˲�������
    * \note ֻ�д˺����ɹ�ִ�к�ſ���������
    */
    void Init(S32 Width,S32 Height,S32 Step,S32 FilterLength);
    /**
    * \brief ��������ͼ���Hessian��Laplacianֵ
    * \param Img �����������ͼ��
    * \note ֻ�д˺����ɹ�ִ�к�ſ���������
    */
    void Calc(const Matrix& Img);
    /**
    * \brief ����ָ��λ�õ�Hessianֵ
    * \param RowPos ��λ��
    * \param ColPos ��λ��
    */
    F32 GetHessian(S32 RowPos,S32 ColPos) const;
    /**
    * \brief ����ָ��λ�õ�Hessianֵ(��ָ���߶���)
    * \param RowPos ��λ��
    * \param ColPos ��λ��
    * \param Scale ָ���ĳ߶�
    */
    F32 GetHessian(S32 RowPos,S32 ColPos,BoxHessian* Scale) const;
    /**
    * \brief ����ָ��λ�õ�Laplacianֵ
    * \param RowPos ��λ��
    * \param ColPos ��λ��
    */
    U8 GetLaplacian(S32 RowPos,S32 ColPos) const;
    /**
    * \brief ����ָ��λ�õ�Laplacianֵ(��ָ���߶���)
    * \param RowPos ��λ��
    * \param ColPos ��λ��
    * \param Scale ָ���ĳ߶�
    */
    U8 GetLaplacian(S32 RowPos,S32 ColPos,BoxHessian* Scale) const;
    /**
    * \brief �˲�����ֵ
    */
    S32 Step;
    /**
    * \brief �˲�������
    */
    S32 FilterLength;

    F32* Hessian;
    U8* Laplacian;
private:
    S32 Width,Height;
    //F32* Hessian;
    //U8* Laplacian;
};
}

#endif // BOXHESSIAN_H
