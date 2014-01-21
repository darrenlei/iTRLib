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
 * calculate.cc
 *  Created on: 2013-9-9
 *      Author: ZYC
 */

#ifndef CALCULATE_H_
#define CALCULATE_H_

#include "../platform/platform.h"
#include "math.h"

namespace itr_math
{
class Calculate
{
public:
    Calculate();
    virtual ~Calculate();
    virtual void Add(S16* SourceA, S16* SourceB, S32 Length, S16* Result) const;
    virtual void Add(S32* SourceA, S32* SourceB, S32 Length, S32* Result) const;
    virtual void Add(F32* SourceA, F32* SourceB, S32 Length, F32* Result) const;

    virtual void Sub(S16* SourceA, S16* SourceB, S32 Length, S16* Result) const;
    virtual void Sub(S32* SourceA, S32* SourceB, S32 Length, S32* Result) const;
    virtual void Sub(F32* SourceA, F32* SourceB, S32 Length, F32* Result) const;

    virtual void Multi(S16* SourceA, S16* SourceB, S32 Length, S16* Result) const;
    virtual void Multi(S32* SourceA, S32* SourceB, S32 Length, S32* Result) const;
    virtual void Multi(F32* SourceA, F32* SourceB, S32 Length, F32* Result) const;

    virtual void Div(S16* SourceA, S16* SourceB, S32 Length, S16* Result) const;
    virtual void Div(S32* SourceA, S32* SourceB, S32 Length, S32* Result) const;
    virtual void Div(F32* SourceA, F32* SourceB, S32 Length, F32* Result) const;

    virtual void DivOrder(S16* SourceA, S32 DivOrder, S32 Length, S16* Result) const;
    virtual void DivOrder(S32* SourceA, S32 DivOrder, S32 Length, S32* Result) const;

    virtual void Offset(S16* SourceA, S16 Offset, S32 Length, S16* Result) const;
    virtual void Offset(S32* SourceA, S32 Offset, S32 Length, S32* Result) const;
    virtual void Offset(F32* SourceA, F32 Offset, S32 Length, F32* Result) const;

    virtual void Scale(S16* SourceA, S16 Scale, S32 Length, S16* Result) const;
    virtual void Scale(S32* SourceA, S32 Scale, S32 Length, S32* Result) const;
    virtual void Scale(F32* SourceA, F32 Scale, S32 Length, F32* Result) const;

    virtual void Normalization(S16* SourceA, S32 Length, S16* Result) const;
    virtual void Normalization(S32* SourceA, S32 Length, S32* Result) const;
    virtual void Normalization(F32* SourceA, S32 Length, F32* Result) const;

    virtual void Abs(S16* SourceA, S32 Length, S16* Result) const;
    virtual void Abs(S32* SourceA, S32 Length, S32* Result) const;
    virtual void Abs(F32* SourceA, S32 Length, F32* Result) const;

    virtual void Opposite(S16* SourceA, S32 Length, S16* Result) const;
    virtual void Opposite(S32* SourceA, S32 Length, S32* Result) const;
    virtual void Opposite(F32* SourceA, S32 Length, F32* Result) const;

    virtual void AddSum(S16* SourceA, S32 Length, S16& Result) const;
    virtual void AddSum(S32* SourceA, S32 Length, S32& Result) const;
    virtual void AddSum(F32* SourceA, S32 Length, F32& Result) const;

    virtual void MultiSum(S16* SourceA, S16* SourceB, S32 Length, S16& Result) const;
    virtual void MultiSum(S16* SourceA, F32* SourceB, S32 Length, S16& Result) const;
    virtual void MultiSum(S32* SourceA, S32* SourceB, S32 Length, S32& Result) const;
    virtual void MultiSum(F32* SourceA, F32* SourceB, S32 Length, F32& Result) const;

    virtual void Product(S16* SourceA, S32 Length, S16& Result) const;
    virtual void Product(S32* SourceA, S32 Length, S32& Result) const;
    virtual void Product(F32* SourceA, S32 Length, F32& Result) const;

    virtual void Set(S16* SourceA, S16 Value, S32 Length) const;
    virtual void Set(S32* SourceA, S32 Value, S32 Length) const;
    virtual void Set(F32* SourceA, F32 Value, S32 Length) const;

    virtual void Max(S16* SourceA, S32 Length, S16& Result) const;
    virtual void Max(S32* SourceA, S32 Length, S32& Result) const;
    virtual void Max(F32* SourceA, S32 Length, F32& Result) const;

    virtual void Max(S16* SourceA, S32 Length, S16& Result) const;
    virtual void Max(S32* SourceA, S32 Length, S32& Result) const;
    virtual void Max(F32* SourceA, S32 Length, F32& Result) const;

    virtual void Compare(F32* SourceA, F32* SourceB, F32 Error,S32 Length,BOOL* IsSame) const;
};
} // namespace itr_math

#endif //* CALCULATE_H_
