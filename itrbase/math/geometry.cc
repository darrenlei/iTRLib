
#include "../platform/platform.h"
#include "math.h"

namespace itr_math
{
//**********Distance2D**********
void Distance2D::SetDXDY(F32 DX, F32 DY)
{
    this->DX = DX;
    this->DY = DY;
}

void Distance2D::SetAngleDistance(F32 Angle, F32 Distance)
{
    F32 sin;
    F32 cos;
    NumericalObj->SinCos(ANG2RAD(Angle), sin, cos);
    this->DX = cos * Distance;
    this->DY = sin * Distance;
}

Distance2D::Distance2D()
{
    SetDXDY(0, 0);
}

Distance2D::Distance2D(const Distance2D& Dis)
{
    SetDXDY(Dis.DX, Dis.DY);
}

void Distance2D::operator=(Distance2D Dis)
{
    SetDXDY(Dis.DX, Dis.DY);
}

Distance2D Distance2D::operator+(Distance2D Dis) const
{
    Distance2D dis;
    dis.SetDXDY(this->DX + Dis.DX, this->DY + Dis.DY);
    return dis;
}

Distance2D Distance2D::operator-(Distance2D Dis) const
{
    Distance2D dis;
    dis.SetDXDY(this->DX - Dis.DX, this->DY - Dis.DY);
    return dis;
}

F32 Distance2D::GetAngle() const
{
    F32 rad;
    NumericalObj->Atan2(DY, DX, rad);
    return RAD2ANG(rad);
}

F32 Distance2D::GetDistance() const
{
    F32 dis = GET_SQUARE(DX) + GET_SQUARE(DY);
    NumericalObj->Sqrt(dis, dis);
    return dis;
}

//**********Point2D**********
void Point2D::SetXY(F32 X, F32 Y)
{
    this->X = X;
    this->Y = Y;
}

Point2D::Point2D()
{
    SetXY(0, 0);
}

Point2D::Point2D(F32 X, F32 Y)
{
    SetXY(X, Y);
}

Point2D::Point2D(const Point2D& Point)
{
    SetXY(Point.X, Point.Y);
}

void Point2D::operator=(const Point2D& Point)
{
    SetXY(Point.X, Point.Y);
}

Distance2D Point2D::operator-(const Point2D& Point) const
{
    Distance2D dis;
    dis.SetDXDY(this->X - Point.X, this->Y - Point.Y);
    return dis;
}

Point2D Point2D::operator+(const Distance2D Dis) const
{
    Point2D point(this->X + Dis.DX, this->Y + Dis.DY);
    return point;
}

Point2D Point2D::operator-(const Distance2D Dis) const
{
    Point2D point(this->X - Dis.DX, this->Y - Dis.DY);
    return point;
}

//**********Point3D**********
Point3D::Point3D(F32 X, F32 Y, F32 Z)
{
    this->X = X;
    this->Y = Y;
    this->Z = Z;
}

//**********RectangleF**********
RectangleF::RectangleF()
{
    this->X = 0;
    this->Y = 0;
    this->Width = 0;
    this->Height = 0;
}
RectangleF::RectangleF(F32 X, F32 Y, F32 Width, F32 Height)
{
    Init(X,Y,Width,Height);
}
void RectangleF::Init(F32 X, F32 Y, F32 Width, F32 Height)
{
    this->X = X;
    this->Y = Y;
    this->Width = Width;
    this->Height = Height;
}
void RectangleF::SetPoint(Point2D P)
{
    X = P.X;
    Y = P.Y;
}

Point2D RectangleF::GetPoint() const
{
    Point2D point(X, Y);
    return point;
}

Point2D RectangleF::GetCenter() const
{
    Point2D point(X + Width * 0.5, Y + Height * 0.5);
    return point;
}

BOOL RectangleF::IsInRectangle(Point2D Point) const
{
    return (IS_IN_RANGE(Point.X, X, X + Width) && IS_IN_RANGE(Point.Y, Y, Y + Height));
}

//**********RectangleS**********
RectangleS::RectangleS()
{
    this->X = 0;
    this->Y = 0;
    this->Width = 0;
    this->Height = 0;

    this->_x = 0;
    this->_y = 0;
    this->_xEnd = 0;
    this->_yEnd = 0;
}
RectangleS::RectangleS(S32 X, S32 Y, S32 Width, S32 Height)
{
    Init(X,Y,Width,Height);
}
void RectangleS::Init(S32 X, S32 Y, S32 Width, S32 Height)
{
    this->X = X;
    this->Y = Y;
    this->Width = Width;
    this->Height = Height;

    this->_x = 0;
    this->_y = 0;
    this->_xEnd = 0;
    this->_yEnd = 0;
}
BOOL RectangleS::IsInRectangle(S32 X, S32 Y) const
{
    return (IS_IN_RANGE(X, this->X, this->X + this->Width)
            && IS_IN_RANGE(Y, this->Y, this->Y + this->Height));
}

void RectangleS::Reset()
{
    _x = X;
    _y = Y;
    _xEnd = X + Width;
    _yEnd = Y + Height;
}

BOOL RectangleS::Next(S32& X, S32& Y)
{
    BOOL r = (_x < _xEnd) && (_y < _yEnd);
    X = _x;
    Y = _y;
    _x++;
    if (_x >= _xEnd)
    {
        _x = X;
        _y++;
    }
    return r;
}
}

