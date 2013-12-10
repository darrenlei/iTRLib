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
 * draw.h
 *  Created on: 2013-10-9
 *      Author: ghdawn
 */

#ifndef DRAW_H_
#define DRAW_H_
#include "../itrvision.h"
#include "../feature/featurepoint.h"
#include "itrbase.h"
#include <vector>
using std::vector;
using itr_math::RectangleS;
namespace itr_vision
{

    class Draw
    {
        public:
            static void Circle(ImageARGB& Img, S32 x, S32 y, S32 r, U32 color);
            static void Circle(ImageGray& Img, S32 x, S32 y, S32 r, S16 color);
            static void LineOffset(ImageGray& Img,S32 x, S32 y, S32 offsetx ,S32 offsety,S16 color);
            static void LineOffset(ImageARGB& Img,S32 x, S32 y, S32 offsetx ,S32 offsety,U32 color);
            static void Line(ImageGray& Img,S32 beginx, S32 beginy, S32 endx ,S32 endy,S16 color);
            static void Line(ImageARGB& Img,S32 beginx, S32 beginy, S32 endx ,S32 endy,U32 color);
            static void Cross(ImageGray &bmp, S32 x, S32 y, S32 scale, S16 color);
            static void Cross(ImageARGB &bmp, S32 x, S32 y, S32 scale, U32 color);
            static void Rectangle(ImageGray& Img,RectangleS rect,S16 color);
            static void Rectangle(ImageARGB& Img,RectangleS rect,U32 color);
            static void Correspond(const ImageGray& Img1,const ImageGray& Img2,const vector<FeaturePoint>& feature1,const vector<FeaturePoint>& feature2,ImageGray& Result);
    };

} // namespace itr_vision
#endif // DRAW_H_
