#include <iostream>
#include <stdio.h>
#include "itrbase.h"
#include "itrvision.h"
#include "joysticktest.h"
#include "asicameratest.h"
#include "V4Ltest.h"
#include "joystickprocesstest.h"
#include "v4ltestbygqp.h"

#include "itrdevice.h"
int main()
{
    //JoyStickTest();
    //asicamera_test();
//    JoyStickProcessTest();
    itr_math::MathObjStandInit();
    //F4Ltest();
     v4ltestguan();
    return 0;
}
