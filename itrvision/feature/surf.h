#ifndef SURF_H
#define SURF_H

#include "itrbase.h"
#include "vectorfeaturepoint.h"
#include <vector>


using namespace itr_math;

namespace itr_vision
{
    /**
    * /brief ����ʵ����SURF�㷨
    * /note ͼ�����ݻ���itr_math::Matrix����,ȫ��Ϊ�����ȸ������.
    * /note ������̾�������itr_math::Calculate�еķ���,�Ӷ�ʵ���˼��㷽ʽ�Ŀ�����Ż�.
    * /note ���ڻῼ�ǽ��ж�ARM-Neon�Ļ���Ż�.
    */
    class SURF
    {
        public:
            /**
            * /brief ��ɶ������������Init()������������.
            */
            SURF();
            /**
            * /brief �ͷŶ�̬������ڴ�
            */
            virtual ~SURF();
        protected:
        private:
        void getOrientation();
        void getDescriptor(bool bUpright = false);
        inline float gaussian(int x, int y, float sig);
        inline float gaussian(float x, float y, float sig);
        inline float haarX(int row, int column, int size);
        inline float haarY(int row, int column, int size);
        float getAngle(float X, float Y);

        Matrix Input;
        VectorFeaturePoint ipts;
        int index;
    };
}
#endif // SURF_H
