#include "joysticktest.h"
#include <stdio.h>

void JoyStickTest()
{
    itr_device::StdJoyStick Joyst;
    char file_name[] ={"/dev/input/js1"};
    if(Joyst.Open(file_name))
    {
        S32 axiscount=0;
        S32 buttoncount=0;
        F32 axisvalue[30]={0};
        S32 buttonstatus[30]={0};

        while(1)
        {
            Joyst.Update();
            usleep(1000);
            axiscount=Joyst.GetAxisCount();
            for(S32 i=0; i<axiscount; i++)
            {
                axisvalue[i]=Joyst.GetAxisValue(i);
            }
            buttoncount=Joyst.GetButtonCount();
            for(S32 i=0; i<buttoncount; i++)
            {
                buttonstatus[i]=Joyst.GetButtonStatus(i);
            }


            printf("\naxis count :%d\naxis value:",axiscount);
            for(S32 i=0; i<axiscount; i++)
            {
                printf("%f\t",axisvalue[i]);
            }
            printf("\nbutton count:%d\n",buttoncount);
            for(S32 i=0; i<buttoncount; i++)
            {
                printf("%d\t",buttonstatus[i]);
            }
        }
    }
    else
        printf("fail to open!\n");

}
