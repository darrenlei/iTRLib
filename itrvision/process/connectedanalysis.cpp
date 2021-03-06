#include "connectedanalysis.h"
#include "itrbase.h"
#include <vector>
#include "../feature/block.h"
#include<algorithm>
#include<queue>
#include <string.h>
using namespace std;
using itr_math::Matrix;
namespace itr_vision
{
ConnectedAnalysis::ConnectedAnalysis()
{

}
ConnectedAnalysis::~ConnectedAnalysis()
{

}
//Do the Contour Analysis work,i.e. get the information of each block, including x,y & area, etc.
void ConnectedAnalysis::Contour(const Matrix &input,vector<Block> &blocks)
{
    this->BNum = 0;
    this->ImgWidth = input.GetCol();
    this->ImgHeight = input.GetRow();
    bool visited[1000*1000];
    memset(visited,0,sizeof(visited));
    int x,y;
    for(int i=0; i<ImgHeight; i++)
    {
        for(int j=0; j<ImgWidth; j++)
        {
            if((!visited[i*ImgWidth+j])&& (input(i,j)>0))
            {
                Block blk;
                //blk.x = x;
                //blk.y = y;
                //blk.Area = 0;//Area = 0 (广度)Area = 1 (深度)
                //Fill(input,x,y,blk,visited);
                Fill_q(input,j,i,blk,visited);
                blk.x /= blk.Area;
                blk.y /= blk.Area;
                blocks.push_back(blk);
                BNum++;
            }
        }
    }
    if(BNum>0)
    {
        SortBlocks(blocks);
        return;
    }

}
//Fill the connected pixel to form a block
void ConnectedAnalysis::Fill(const Matrix &input,S32 x,S32 y,Block& blk,bool* visited)
{
    const int dx[] = {-1,0,1,-1,1,-1,0,1};
    const int dy[] = {-1,-1,-1,0,0,1,1,1};
    visited[y*ImgWidth+x] = 1;
    for(int i =0; i<8; i++)
    {
        int newx = x + dx[i];
        int newy = y + dy[i];
        if(newx < 0|| newx >= ImgWidth)continue;
        if(newy < 0|| newy >= ImgHeight)continue;
        if(PixEql(input(y,x),input(newy,newx)) && visited[newy*ImgWidth+newx]==0)
        {
            blk.Area++;
            blk.x += newx;
            blk.y += newy;
            Fill(input,newx,newy,blk,visited);
        }
    }
}
void ConnectedAnalysis::Fill_q(const Matrix &input,S32 x,S32 y,Block& blk,bool* visited)
{
    queue< pair<S32,S32> > unvisited;
    pair<S32,S32> current;
    pair<S32,S32> temp (x,y);
    S32 newx = 0;
    S32 newy = 0;
    F32 value=input(y,x);
    const int dx[] = {-1,0,1,-1,1,-1,0,1};
    const int dy[] = {-1,-1,-1,0,0,1,1,1};

    unvisited.push(temp);
    visited[temp.second*ImgWidth+temp.first] = 1;
    while(!unvisited.empty())
    {
        current = (unvisited.front());
        for(S32 i=0; i<8; i++)
        {
            newx = current.first + dx[i];
            newy = current.second + dy[i];
            if(visited[newy*ImgWidth+newx]==0&&PixIn(newx,newy) && PixEql(value,input(newy,newx)) )
            {
                temp.first = newx;
                temp.second = newy;
                unvisited.push(temp);
                visited[temp.second*ImgWidth+temp.first] = 1;
            }
        }
        blk.Area++;
        blk.x += current.first;
        blk.y += current.second;
        unvisited.pop();
    }
}

//Sort the block
int BlockCmp(Block a,Block b)
{
    return a.Area > b.Area;
}
void ConnectedAnalysis::SortBlocks(vector<Block> &blocks)
{
    sort(blocks.begin(),blocks.end(),BlockCmp);
}
}
