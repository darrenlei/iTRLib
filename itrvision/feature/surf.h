#ifndef SURF_H
#define SURF_H

#include "itrbase.h"

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
    };
}
#endif // SURF_H
