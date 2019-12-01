#include <iostream>
#include <armadillo>
#include <sstream>


int main()
{

std::unordered_map<std::string, double> umap;

arma::umat X = { 
        {0,    1,    1,    0,    0,    0,    0,    0,    0,     0},
        {0,    0,    1,    1,    0,    0,    0,    0,    0,     0},
        {0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
        {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
        {1,    0,    0,    0,    0,    0,    1,    0,    0,     0},
        {0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
        {0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
        {1,    0,    0,    0,    0,    0,    0,    0,    0,     0},
        {0,    0,    0,    0,    0,    0,    0,    0,    0,     1},
        {0,    0,    0,    0,    1,    1,    0,    0,    0,     0}
           };

arma::ivec pos1 = {2,1};// (x,y)

arma::cout<<X<<arma::endl<<pos1<<arma::endl;

std::ostringstream s;

s<<X<<pos1;
std::string key1=s.str();

s.str(""); //clear out
X(0,0)=1;
s<<X<<pos1;
std::string key2=s.str();

//std::cout<<"here is me=="<<std::endl<<key1<<std::endl;
umap[key1] = 1.234;

//std::cout<<"here is me2=="<<std::endl<<key2<<std::endl;

umap[key2] = 1.2345;


// Get an iterator pointing to begining of map
std::unordered_map<std::string, double>::iterator it = umap.begin();
 
// Iterate over the map using iterator
while(it != umap.end())
{
    std::cout<<it->first << " :: "<<it->second<<std::endl;
    it++;
}



    return 0;
}



