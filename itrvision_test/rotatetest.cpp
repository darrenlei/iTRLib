#include "rotatetest.h"
#include "iohelper.h"
#include "itrvision.h"
#include "itrbase.h"
#include <math.h>

void rotatetest()
{
    rotateImgGraytest();
}

void rotateImgGraytest()
{
    ImageGray grayI;
    ImageGray grayO(100, 100);
    RectangleS rect(100, 100, 100, 100);
    F32 ang = 90;
    IOHelper::ReadPGMFile("Debug/table1.pgm", grayI);
    Rotate::rotate(grayI, rect, ang, grayO);
    IOHelper::WritePGMFile("Debug/grayO.pgm", grayO);
}
