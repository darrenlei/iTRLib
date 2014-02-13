#include "vectorfeaturepoint.h"

#include "itrbase.h"

using namespace itr_math;

namespace itr_vision
{

    VectorFeaturePoint::VectorFeaturePoint()
    {
        //ctor
        this->LevelNo =0;
        this->ID =0;
        this->Tag =NULL;
        this->Dir =0;
        this->Quality=0;
        this->Value=0;
    }

    VectorFeaturePoint::~VectorFeaturePoint()
    {
        //dtor
    }

    VectorFeaturePoint::VectorFeaturePoint(const VectorFeaturePoint &other)
    {
        //copy ctor
        this->LevelNo =other.LevelNo;
        this->ID =other.ID;
        this->Tag =other.Tag;
        this->Dir =other.Dir;
        this->Quality=other.Quality;
        this->Value=other.Value;
    }
    void VectorFeaturePoint::Init(Point2D Pos,S32 ID,F32 Dir,F32 Quality,F32 Value,S32 FeatureDim,S32 LevelNo)
    {
        this->SetXY(Pos.X, Pos.Y);
        this->ID =ID;
        this->Dir=Dir;
        this->Quality=Quality;
        this->Value=Value;
        this->Feature.Init(FeatureDim);
        this->LevelNo=LevelNo;
    }
    void VectorFeaturePoint::Init(Point2D Pos,F32 Dir,S32 FeatureDim,S32 LevelNo)
    {
        this->SetXY(Pos.X, Pos.Y);
        this->Dir=Dir;
        this->Feature.Init(FeatureDim);
        this->LevelNo=LevelNo;
    }
}