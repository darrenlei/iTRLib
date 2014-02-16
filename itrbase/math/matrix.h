/*
 *
 *   Copyright (C) 2013 BUAA iTR Research Center. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * matrix.h
 *  Created on: 2013-9-14
 *      Author: zhouyi
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "../platform/platform.h"
#include "math.h"
#include <stdio.h>

namespace itr_math
{
/*
 * 矩阵所有操作参数均为先行号(Row=Y=Height)后列号(Col=X=Width)
 * 矩阵初始化只支持以下几种形式，不支持初始化直接等于另一个矩阵。
 * eg:Matrix a = Source (不允许这样用)
 */
class Matrix
{
public:
    //**********构造&析构**********
    /*
     * 初始化一个指定行列数的空矩阵(自动分配内存)
     */
    Matrix(S32 Row, S32 Col);
    /*
     * 初始化一个指定行列数的矩阵(以传入的指针为数据区,不分配本地内存)
     */
    Matrix(S32 Row, S32 Col, F32* Data);
    /*
     * 初始化一个指定阶数的方阵(自动分配内存)
     */
    Matrix(S32 RowCol);
    /*
     * 初始化完全一样的矩阵(Clone)
     */
    Matrix(const Matrix& Mat);
    /*
     * 用于初始化列表的空构造函数
     * 在构造后需手动调用Init函数
     */
    Matrix();
    /*
     * 回收自动分配的内存
     */
    virtual ~Matrix();

    //**********后初始化函数**********
    /*
     * 初始化一个指定行列数的空矩阵(自动分配内存，只能在无参数构造对象后调用)
     */
    void Init(S32 Row, S32 Col);
    /*
     * 初始化一个指定行列数的矩阵(以传入的指针为数据区，只能在无参数构造对象后调用)
     */
    void Init(S32 Row, S32 Col, F32* Data);
    /*
     * 初始化一个指定阶数的方阵(自动分配内存，只能在无参数构造对象后调用)
     */
    void Init(S32 RowCol);
    //**********初等变换**********
    /*
     * 将RowPosSource行加RowPosTarget行
     */
    void virtual AddRow(S32 RowPosSource, S32 RowPosTarget);
    /*
     * 将Data加至RowPosTarget行
     */
    void virtual AddRow(F32* Data, S32 RowPosTarget);
    /*
     * 将RowPosSource行减至RowPosTarget行
     */
    void virtual SubRow(S32 RowPosSource, S32 RowPosTarget);
    /*
     * 将Data减至RowNoTarget行
     */
    void virtual SubRow(F32* Data, S32 RowPosTarget);
    /*
     * 将RowNoTarget行乘以K
     */
    void virtual MulRow(F32 K, S32 RowPosTarget);
    //Swap Row
    /*
     * 交换RowNoA行和RowNoB行
     */
    void virtual SwapRow(S32 RowNoA, S32 RowNoB);
    /*
     * 将ColNoAdd列加至ColNoTarget列
     */
    void virtual AddCol(S32 ColPosSource, S32 ColPosTarget);
    /*
     * 将Data加至ColNoTarget列
     */
    void virtual AddCol(F32* Data, S32 ColPosTarget);
    /*
     * 将ColNoSub列减至ColNoTarget列
     */
    void virtual SubCol(S32 ColPosSource, S32 ColPosTarget);
    /*
     * 将Data减至ColNoTarget行
     */
    void virtual SubCol(F32* Data, S32 ColPosTarget);
    /*
     * 将ColNoTarget列乘以K
     */
    void virtual MulCol(F32 K, S32 ColPosTarget);
    /*
     * 交换ColNoA列和ColNoB列
     */
    void virtual SwapCol(S32 ColNoA, S32 ColNoB);

    //**********常量相关计算**********
    /*
     * 全部元素加上K
     */
    void virtual AllAdd(F32 K);
    /*
     * 全部元素乘以K
     */
    void virtual AllMul(F32 K);

    //**********矩阵相关计算**********

    /*
     * 加上矩阵MatrixAdd
     */
    void virtual Add(const Matrix& Mat);
    /*
     * 减去矩阵MatrixSub
     */
    void virtual Sub(const Matrix& Mat);
    /*
     * 右乘矩阵Mat并将结果存至MatResult
     */
    void virtual Mul(const Matrix& Mat, Matrix& MatResult) const;

    Matrix virtual operator*(const Matrix& Mat) const;
    Matrix virtual operator+(const Matrix& Mat) const;
    Matrix virtual operator-(const Matrix& Mat) const;
    Matrix virtual operator=(const Matrix& Mat);
    //**********向量相关计算**********
    Vector virtual operator*(const Vector& vec) const;
    //**********常用操作**********
    /*
     * 用于抽取矩阵中的某列
     */
    void virtual ColExtract(F32* Data, S32 Offset, S32 Interval, S32 Length, F32* Result);
    /******************************
    *******Advanced Function*******
    *******************************/

    /*
     * 求矩阵逆并将结果放至MatResult
     */
    BOOL virtual Inv(Matrix& MatResult) const;
    /*
     * 求矩阵逆
     */
    Matrix virtual Inv() const;
    /*
     * 求矩阵转置并将结果放至MatResult
     */
    void virtual Tran(Matrix& MatResult) const;
    /*
    * 求矩阵转置
    */
    Matrix virtual Tran() const;
    /*
    * 求矩阵tr
    */
    void virtual Tr(F32 K)
    {
        K=0;
        for (S32 i = 0; i < row&&i<col; i++)
            K+=data[i * row + i];
    }
    //**********内联函数**********
    /*
     * 获取行数
     */
    inline virtual S32 GetRow() const
    {
        return row;
    }
    /*
     * 获取列数
     */
    inline virtual S32 GetCol() const
    {
        return col;
    }
    /*
     * 获取数据地址
     */
    inline virtual F32* GetData() const
    {
        return data;
    }
    /*
     * 是否为初始化自动分配的本地数据
     */
    inline virtual BOOL IsLocalData() const
    {
        return localData;
    }

    //**********数据转移**********
    /*
     * Func:将传入的数据复制至指定的矩形区域
     * Para:RowOffset,ColOffset:插入位置
     *      Width,Height:插入的矩形行区域尺寸
     */
    inline void virtual CopyFrom(S32 RowPos, S32 ColPos, S32 Width, S32 Height, F32* Data)
    {
        assert(RowPos >= 0 && RowPos < row);
        assert(ColPos >= 0 && ColPos < col);
        assert(RowPos + Height <= row);
        assert(ColPos + Width <= col);
        assert(Data!=NULL);
        for (S32 i = 0; i < Height; i++)
        {
            MemoryCopy(data + RowPos  * (col+i) + ColPos, Data + i * Width, Width * sizeof(F32));
        }
    }
    /*
     * Func:将传入的数据全部复制到矩阵中
     * Para:
     */
    inline void virtual CopyFrom(F32* Data)
    {
        assert(Data!=NULL);
        MemoryCopy(data, Data, row * col * sizeof(F32));
    }
    /*
     * Func:将指定的矩形区域复制出来
     * Para:RowPos,ColPos:取出的位置
     *      Width,Height:矩形尺寸
     */
    inline void virtual CopyTo(S32 RowPos, S32 ColPos, S32 Width, S32 Height,
                               F32* Data) const
    {
        assert(RowPos >= 0 && RowPos < row);
        assert(ColPos >= 0 && ColPos < col);
        assert(RowPos + Height <= row);
        assert(ColPos + Width <= col);
        assert(Data!=NULL);
        for (S32 i = 0; i < Height; i++)
        {
            MemoryCopy(Data + i * Width, data + RowPos  * (col+i) + ColPos, Width * sizeof(F32));
        }
    }
    /*
     * 将全部数据复制出来
     */
    inline void virtual CopyTo(F32* Data) const
    {
        assert(Data!=NULL);
        MemoryCopy(Data, data, row * col * sizeof(F32));
    }

    //Copy Row From
    /*
    * 将传入数据复制到指定行的部分区域
    */
    inline void virtual CopyRowFrom(S32 RowPos, S32 ColPos, S32 ColNum, F32* Data)
    {
        assert(RowPos >= 0 && RowPos < row);
        assert(ColPos >= 0 && ColPos < col);
        assert(ColNum >= 0 && ColNum + ColPos <= col);
        assert(Data!=NULL);
        MemoryCopy(data + RowPos * col + ColPos , Data, ColNum * sizeof(F32));
    }
    /*
     * 将传入数据复制到指定行
     */
    inline void virtual CopyRowFrom(S32 RowPos, F32* Data)
    {
        assert(RowPos >= 0);
        assert(RowPos < row);
        assert(Data!=NULL);
        MemoryCopy(data + RowPos * col, Data, col * sizeof(F32));
    }
    /*
     * 复制指定行的部分数据出来
     */
    inline void virtual CopyRowTo(S32 RowPos, S32 ColPos, S32 ColNum, F32* Data) const
    {
        assert(RowPos >= 0 && RowPos < row);
        assert(ColPos >= 0 && ColPos < col);
        assert(ColNum + ColPos <= col);
        assert(Data !=NULL);
        MemoryCopy(Data, data + RowPos * col + ColPos, ColNum * sizeof(F32));
    }
    /*
     * 复制指定行数据处理
     */
    inline void virtual CopyRowTo(S32 RowPos, F32* Data) const
    {
        assert(RowPos >= 0);
        assert(RowPos < row);
        assert(Data!=NULL);
        MemoryCopy(Data, data + RowPos * col, col * sizeof(F32));
    }
    /*
     * 将数据复制到指定列的部分区域
     */
    inline void virtual CopyColFrom(S32 ColPos, S32 RowPos, S32 RowNum, F32* Data)
    {
        assert(ColPos >= 0 && ColPos < col);
        assert(RowPos >= 0 && RowPos < row);
        assert(RowPos + RowNum <= row);
        assert(Data!=NULL);
        S32 pos=RowPos*col+ColPos;
        for (S32 i = 0; i < RowNum; i++)
        {
            data[pos]=Data[i];
            pos+=col;
        }
    }
    /*
     * 将数据复制到指定列
     */
    inline void virtual CopyColFrom(S32 ColPos, F32* Data)
    {
        assert(ColPos >= 0 && ColPos < col);
        assert(Data != NULL);
        S32 pos=ColPos;
        for (S32 i = 0; i < row; i++)
        {
            data[pos]=Data[i];
            pos+=col;
        }
    }
    /*
    * 复制指定列的部分区域数据出来
    */
    inline void virtual CopyColTo(S32 ColPos, S32 RowPos, S32 RowNum, F32* Data) const
    {
        assert(ColPos >= 0 && ColPos < col);
        assert(RowPos >= 0 && RowPos < row);
        assert(RowPos + RowNum <= row);
        assert(Data!=NULL);
        S32 pos=RowPos*col+ColPos;
        for (S32 i = 0; i < RowNum; i++)
        {
            Data[i]=data[pos];
            pos+=col;
        }
    }
    /*
     * 复制指定列的数据出来
     */
    inline void virtual CopyColTo(S32 ColPos, F32* Data) const
    {
        assert(ColPos >= 0 && ColPos < col);
        assert(Data != NULL);
        S32 pos=ColPos;
        for (S32 i = 0; i < row; i++)
        {
            Data[i]=data[pos];
            pos+=col;
        }
    }

    //**********数据访问**********
    /*
     * 写入单个元素(一维线性访问)
     */
    inline F32& operator[](S32 index)
    {
        assert(index >=0);
        assert(index < row * col);
        return data[index];
    }
    /*
     * 读取单个元素(一维线性访问)
     */
    inline F32 operator[](S32 index) const
    {
        assert(index >=0);
        assert(index < row * col);
        return data[index];
    }
    /*
     * 写入单个元素(Y=行数,X=列数),且会自动执行限位。
     */
    inline F32& operator()(S32 Y, S32 X)
    {
        if(X<0)
            X=0;
        if(X>=col)
            X=col-1;
        if(Y<0)
            Y=0;
        if(Y>=row)
            Y=row-1;
        return data[Y * col + X];
    }
    /*
     * 读取单个元素(Y=行数,X=列数),且会自动执行限位。
     */
    inline F32 operator()(S32 Y, S32 X) const
    {
        if(X<0)
            X=0;
        if(X>=col)
            X=col-1;
        if(Y<0)
            Y=0;
        if(Y>=row)
            Y=row-1;
        return data[Y * col + X];
    }

    //**********数据操作**********
    /*
     * 设置所有元素为K
     */
    inline void virtual Set(F32 K)
    {
        for (S32 i = 0; i < row * col; i++)
            data[i] = K;
    }
    /*
     * 设置所有元素为0
     */
    inline void virtual Clear()
    {
        Set(0);
    }
    /*
     * 设置主对角线元素为K
     */
    inline void virtual SetDiag(F32 K)
    {
        for (S32 i = 0; i < row; i++)
            data[i * row + i] = K;
    }
    /*
     * 设置主对角线元素为Data
     */
    inline void virtual SetDiag(F32* Data)
    {
        assert(Data!=NULL);
        for (S32 i = 0; i < row; i++)
            data[i * row + i] = Data[i];
    }
    /*
     * 将主对角线元素放至Data
     */
    inline void virtual GetDiag(F32* Data) const
    {
        assert(Data!=NULL);
        for (S32 i = 0; i < row; i++)
            Data[i] = data[i * row + i];
    }

    //**********维数匹配**********
    /*
     * 检查维数与Mat是否一致
     */
    inline BOOL virtual MatchDim(const Matrix& Mat) const
    {
        if (Mat.row == row && Mat.col == col)
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否为Row,Col
     */
    inline BOOL virtual MatchDim(S32 Row, S32 Col) const
    {
        if (Row == row && Col == col)
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否可右乘Mat
     */
    inline BOOL virtual MatchMul(const Matrix& Mat) const
    {
        if (col == Mat.row)
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否可右乘行向量Vec
     */
    inline BOOL virtual MatchRightMulRow(const Vector& Vec) const
    {
        if (col == Vec.GetDim())
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否可右乘列向量Vec
     */
    inline BOOL virtual MatchRightMulCol(const Vector& Vec) const
    {
        if (col == 1&&row==Vec.GetDim())
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否可左乘行向量Vec
     */
    inline BOOL virtual MatchLeftMulRow(const Vector& Vec) const
    {
        if (row == Vec.GetDim())
            return true;
        else
            return false;
    }
    /*
     * 检查维数是否可左乘列向量Vec
     */
    inline BOOL virtual MatchLeftMulCol(const Vector& Vec) const
    {
        if (row == 1&&col==Vec.GetDim())
            return true;
        else
            return false;
    }

    inline BOOL CompMatrix(Matrix& Mat)
    {
        BOOL r;
        r=MatchDim(Mat);
        if(r==false)
            return false;
        itr_math::CalculateObj->Compare(GetData(),Mat.GetData(),1,col*row,&r);
        return r;
    }

private:
    S32 row, col;
    F32* data;
    BOOL localData;
};

} // namespace itr_math

#endif // MATRIX_H_
