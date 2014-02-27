#ifndef FORMATCONVERTER_H
#define FORMATCONVERTER_H

#include "itrbase.h"

using itr_math::Matrix;

namespace itr_vision
{

/**
* \brief �������ڽ�����ɼ��ĸ���Raw����ת��
*/
class FormatConverter
{
public:
    enum RawFormat
    {
        RawMono8,
        RawMono16,
        RawYUV444,
        RawYUV422,
        RawYUV411,
        RawYUV420,
        RawRGB24,
        RawRGB48,
        RawBGR24,
        RawRGBA32,
        RawBGRA32,
        RawBayer8RGGB,
        RawBayer8GBRG,
        RawBayer8GRBG,
        RawBayer8BGGR,
        RawBayer16RGGB,
        RawBayer16GBRG,
        RawBayer16GRBG,
        RawBayer16BGGR,
    };
    /**
    * /brief ��ʾͼ��ɫ��ͨ��
    */
    enum ColorChannel
    {
        R,//Red
        G,//Green
        B,//Blue
        A,//Alpha
        Y,//Y
        U,//Cb
        V,//Cr
        M,//Gray
        H,//Hue
        S,//Chroma
        L,//Brightness
    };
    /**
    * /brief �˺������ڽ�Raw��ʽת��ΪMatrix
    * /param Raw ��ת����ԭʼ��������
    * /param Length ��ת����ԭʼ�������ݳ���
    * /param Format ��ת����ԭʼ�������ݸ�ʽ
    * /param Channel ת����Ŀ�����ɫ��ͨ��
    * /param Mat ת������ľ������,����ǰ�����������,ͼ��ת���ߴ��������С.
    * /note ת��������ݷ�ΧΪ0~1
    */
    static bool Raw2Matrix(U8* Raw,S32 Length,RawFormat Format,ColorChannel Channel,Matrix& Mat);
protected:
private:
};
}

#endif // RAWCONVERTER_H
