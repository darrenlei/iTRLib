#ifndef STANDSERIALPROTOCOL_H
#define STANDSERIALPROTOCOL_H

#include "string.h"
#include "../platform/platform.h"
//#include "../math/math.h"
#include "../container/container.h"
#include "../encode/crc.h"

#define StandSerialProtocolMaxLength 255//֡��󳤶�
#define StandSerialProtocolHeadLength 2//��ͷ����
#define StandSerialProtocolMateLength 2//������Ϣ����
#define StandSerialProtocolCRCLength 2//CRC����
#define StandSerialProtocolPlusLength (StandSerialProtocolHeadLength+StandSerialProtocolMateLength+StandSerialProtocolCRCLength)//֡������Ϣ�ܳ�
#define StandSerialProtocolKey1Length 16//��Կ1����
#define StandSerialProtocolKey2Length 4//��Կ2����

namespace itr_protocol
{

/**
* \brief ֡�ṹ����ṹ��
*/
typedef struct
{
    /**
    * \brief ������0
    */
    U8 S0;
    /**
    * \brief ������1
    */
    U8 S1;
    /**
    * \brief ֡���ܳ�
    */
    U8 Length;
    /**
    * \brief ֡������
    */
    U8 Property;
    /**
    * \brief ֡������ʼ
    */
    U8 Package;
    //U16 CRC;

} StandSerialFrameStruct;

/**
* \brief ��׼����Э��֧�ֿ�(SSP)
*/
class StandSerialProtocol
{
public:
    /**
    * \brief ֡���ջص�����ģ��
    * \param SSP ���յ����ݵ�StandSerialProtocol����
    * \param SSFS ���յ���֡�ṹ���ַ
    * \param Package ���յ����ݰ���ַ
    * \param PackageLength ���յ����ݰ�����
    */
    typedef void (*SSPReceive)(StandSerialProtocol* SSP, StandSerialFrameStruct* SSFS,U8* Package,S32 PackageLength);
    /**
    * \brief Ĭ�Ϲ���
    */
    StandSerialProtocol();
    /**
    * \brief Ĭ������
    */
    virtual ~StandSerialProtocol();
    /**
    * \brief Ĭ�ϸ���
    */
    StandSerialProtocol(const StandSerialProtocol& other);
    /**
    * \brief ��ʼ�����趨��������
    * \param S0 �趨��������0
    * \param S1 �趨��������1
    * \param DataSendFun ���ݷ��ͺ���ָ��(�˿����ô˺������ڷ�������)
    */
    void Init(U8 S0, U8 S1, StandDataSendFun DataSendFun);
    /**
    * \brief ����ԭʼ������
    * \param Buffer ����������
    * \param BufferLength ��������������
    */
    S32 ProcessRawByte(U8* Buffer, S32 BufferLength);
    /**
    * \brief ��SSPЭ�鷢��ָ�����ݰ�
    * \param ProtocolID Э��ID
    * \param Package ���������ݰ�
    * \param PackageLength ���������ݰ�����
    */
    void SSPSendPackage(U8 ProtocolID, U8* Package, U8 PackageLength);
    /**
    * \brief ���֡�ܳ�
    * \param SSFS ����֡
    * \return ֡�ܳ�
    */
    static U8 GetSSFSLength(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡����
    * \param SSFS ����֡
    * \return ֡����
    */
    static U8 GetSSFSProperty(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡Э��ID
    * \param SSFS ����֡
    * \return Э��ID
    */
    static U8 GetSSFSProtocolID(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡�Ƿ����
    * \param SSFS ����֡
    * \return ֡�Ƿ����
    */
    static U8 GetSSFSIsEncrypt(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡���ݰ�
    * \param SSFS ����֡
    * \return ֡���ݰ�
    */
    static U8* GetSSFSPackage(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡���ݰ�����
    * \param SSFS ����֡
    * \return ֡���ݰ�����
    */
    static U8 GetSSFSPackageLength(StandSerialFrameStruct* SSFS);
    /**
    * \brief ���֡У��
    * \param SSFS ����֡
    * \return ֡У��
    */
    static U16 GetSSFSCRC(StandSerialFrameStruct* SSFS);
    /**
    * \brief ��Կ1
    */
    U8 Key1[StandSerialProtocolKey1Length];
    /**
    * \brief ��Կ2
    */
    U8 Key2[StandSerialProtocolKey2Length];
    /**
    * \brief ������0
    */
    U8 S0;
    /**
    * \brief ������1
    */
    U8 S1;
    /**
    * \brief �Ƿ��Զ�����
    */
    bool AutoDecrypt;
    /**
    * \brief �Ƿ��Զ�����
    */
    bool AutoEncrypt;
    /**
    * \brief ��Э��ID��Ӧ�Ļص�����ָ����(�յ���Ӧ����֡���Զ�����)
    */
    SSPReceive ProcessFunction[16];
    /**
    * \brief ���ݷ��ͺ���ָ��(�˿����ô˺������ڷ�������)
    */
    StandDataSendFun DataSendFun;
protected:
private:
    /**
    * \brief ���ܱ���(��ʱ��ʵ��)
    */
    void SSPEncrypt(StandSerialFrameStruct* SSP, U8* Key1, U8* Key2);
    /**
    * \brief ���ܱ���(��ʱ��ʵ��)
    */
    void SSPDecrypt(StandSerialFrameStruct* SSP, U8* Key1, U8* Key2);
    /**
    * \brief ��鳤�Ⱥ�CRC�Ƿ�Ϸ�
    */
    bool SSPFormatCheck(StandSerialFrameStruct* SSPF);
    /**
    * \brief ����֡�ܳ�
    * \param SSFS ����֡
    * \param Length �ܳ�
    */
    static void SetSSFSLength(StandSerialFrameStruct* SSFS,U8 Length);
    /**
    * \brief ����֡����
    * \param SSFS ����֡
    * \param Property ����
    */
    static void SetSSFSProperty(StandSerialFrameStruct* SSFS,U8 Property);
    /**
    * \brief ����֡Э��ID
    * \param SSFS ����֡
    * \param Property Э��ID
    */
    static void SetSSFSProtocolID(StandSerialFrameStruct* SSFS,U8 ProtocolID);
    /**
    * \brief ����֡�Ƿ����
    * \param SSFS ����֡
    * \param Property �Ƿ����
    */
    static void SetSSFSIsEncrypt(StandSerialFrameStruct* SSFS,U8 IsEncrypt);
    /**
    * \brief ����֡У��
    * \param SSFS ����֡
    * \param Property У��
    */
    static void SetSSFSCRC(StandSerialFrameStruct* SSFS,U16 CRC);
    /**
    * \brief ���ݽ���״ָ̬ʾ
    */
    S32 receiveBufferState;
    /**
    * \brief ���ݽ��ջ�����
    */
    U8 receiveBuffer[StandSerialProtocolMaxLength];
    /**
    * \brief ���ݷ��ͻ�����
    */
    U8 sendBuffer[StandSerialProtocolMaxLength];
};

}

#endif // STANDSERIALPROTOCOL_H
