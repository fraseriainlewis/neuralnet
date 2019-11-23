// envDAG.cpp
#include "envDAG.hpp"
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <gsl/gsl_math.h>

#define DEBUG

envDAG::envDAG(const std::string datafile, const double _alpha_w, const double _alpha_m): alpha_w(_alpha_w), alpha_m(_alpha_m) { 
	std::cout<<"constructing!"<<std::endl;

    /** firstly read in the raw data from disk - CSV without header line **/
	bool ok = rwdata.load(datafile);

    if(ok == false){
    	std::cout << "problem with loading raw data from file" << std::endl;
    	exit(-1);
    }

    /** compute prior constants used across all model fitting */
    // nu  = ones(1,n);  % prior precision for each node = 1
    // mu0 = zeros(1,n); % prior mu0 for each node = 0
    // b=zeros(n,n);     % this is the regression coefs - all zero 

    n = rwdata.n_cols; //number of variables - dimension of DAG
    N = rwdata.n_rows; //number of obs
    //std::cout<<"n="<<n<<std::endl;
    nu = arma::ones<arma::umat>(1,n);
    mu0 = arma::zeros<arma::mat>(1,n);
    b = arma::zeros<arma::mat>(n,n);
   /* mu0(0,0) = 0.1;
    mu0(0,1) = -0.3;
    mu0(0,2) = 0.2;*/
  
    /*b(0,2)=1;
    b(1,2)=1; */              
   

    // % Compute T = precision matrix in Wishart prior.
    // %%%%%%% equation 5 and 6 in Geiger and Heckerman
    envDAG::setT();
    arma::cout<<"T=\n"<<T<<arma::endl;

    envDAG::setR();
    arma::cout<<"R=\n"<<R<<arma::endl;
    

    /** create and set empy dag **/
    dag0 = arma::zeros<arma::umat>(n,n);
 
    /*dag0(1-1,2-1)=1;dag0(1-1,3-1)=1;
    dag0(2-1,3-1)=1;dag0(2-1,4-1)=1;
    dag0(3-1,4-1)=1;
*/

    dag0(1-1,2-1)=1; dag0(1-1,3-1)=1; dag0(1-1,4-1)=1;
    dag0(2-1,3-1)=1;
    dag0(3-1,4-1)=1;

}

void envDAG::PrintData(void) const {
	arma::cout<<"number of variables="<<rwdata.n_cols<<arma::endl;
}

double envDAG::getAlpha_w(void) const {
    
    std::cout<<"_w="<<alpha_w<<" _m="<<alpha_m<<std::endl;
    
    return alpha_w;
}


void envDAG::setT(void){

 //n=3;//b.n_rows; // number of variables
 arma::uword wsize=1;
 
 arma::mat w1  = arma::zeros<arma::mat>(wsize,wsize);
 arma::mat w2,e1,sigmainv;
 unsigned int i=0;

 w1(0,0) = 1/nu(0,0);
 // w1 is current matrix
 // w2 is new matrix


 for(i=1;i<n;i++){
  	wsize=i;
  	w2  = arma::zeros<arma::mat>(wsize+1,wsize+1); // initialize
  	e1=b.submat(0,i,(i-1),i);// row 0:(i-1), col i 
    //e1=e2;
  	w2.submat(0,0,(wsize-1),(wsize-1)) = w1+kron(e1,e1.t())/nu(0,i); // set submatrix top corner
  	w2.submat(wsize,0,wsize,(wsize-1))= -e1.t()/nu(0,i); //last row
  	w2.submat(0,wsize,(wsize-1),wsize)= -e1/nu(0,i);  // last col
  	w2(wsize,wsize) = 1/nu(0,i); // bottom right cell

  	w1=w2; // update copy new matrix to current matrix
 }

 
 //m = w2; // return the final matrix Wn
 
 sigmainv=w2;
 double sigmaFactor = (alpha_m+1)/(alpha_m*(alpha_w-n-1));
 T =inv(sigmainv)/sigmaFactor;

}


void envDAG::setR(void){
 
 arma::mat xbarL=mean(rwdata);//col - variable - means
 // this part compute the equivalent of and outer multiplication - (mu0-xbarL).*(mu0-xbarL)' in matlab col vec * row vec element at a time
 arma::mat tmp1 = (mu0-xbarL); // row vector
 arma::mat tmp2 = tmp1.t();//col vector
 arma::mat tmp  = arma::zeros<arma::mat>(n,n);
 unsigned int i=0;
 for(i=0;i<n;i++){tmp.col(i)=(tmp1*tmp2(i,0)).t();}

 R = T + cov(rwdata)*(N-1) + (alpha_m*N)/(alpha_m+N) * tmp;


}


void envDAG::fitDAG(const arma::umat dag){

arma::cout<<"this is fitdag"<<arma::endl<<" dag="<<dag<<arma::endl<<"lnscore="<<lnscore<<arma::endl;


}

void envDAG::fitDAG(void){


arma::uword nrow=dag0.n_rows; // n is dimension
double totLogScore=0.0;    
unsigned int i,j;
arma::umat nodei;
arma::uvec par_idx,YY;
arma::uword npars=0;
arma::uword l=0;
double A,B;

for(i=0;i<nrow;i++){
    // process node i
    nodei=dag0.row(i);
    //arma::cout<<"nodei="<<nodei<<arma::endl;
    par_idx=find(nodei);//% indexes of parents of node i
    npars=par_idx.n_elem;
    //std::cout<<"num pars="<<npars<<std::endl;
    if(npars==0){ 
      std::cout<<"no parents"<<std::endl;
      // we are done as p(d) = singleX/1.0
      l=npars+1;//l=dimension of d - note here npars=0 always, just so same line can be used below
      YY.set_size(l);
      YY(0)=i;
      //totLogScore=totLogScore+pDln(N,n,l,alpha_m,alpha_w,T,R,YY);
      std::cout<<"nodescore for node="<<i<<pDln(l,YY)<<"l="<<l<<"YY="<<YY<<std::endl;

      totLogScore = totLogScore + pDln(l,YY);

     } else { // we have parents
     	std::cout<<"parents"<<std::endl;

             l=npars+1;//l=dimension of d - note here npars=0 always, just so same line can be used below
             // we want to do matlab equivalent YY=[par_idx i], where par_idx is a vector - 
             YY.set_size(l);
             for(j=0;j<l-1;j++){YY(j)=par_idx(j);}
             YY(l-1)=i;
             A=pDln(l,YY);   
         // repeat as above but with current node in YY - matlabe equivalent YY=[par_idx];
             l--;//decrement l as one less node in calc
             YY.set_size(l);
             for(j=0;j<l;j++){YY(j)=par_idx(j);}
             B=pDln(l,YY);
             totLogScore=totLogScore+A-B;
         
         }


} // end of for over nodes 



lnscore = totLogScore;

arma::cout<<"this is fitdag"<<arma::endl<<" dag0="<<dag0<<arma::endl<<"lnscore=" << std::setprecision(4) << std::scientific<<lnscore<<arma::endl;

}

double envDAG::pDln(const unsigned int l, const arma::uvec YY)
{
 double term1,term2,term3,topGamma,botGamma,topdet,botdet;

 term1 = (l/2.0)*log(alpha_m/(N+alpha_m));
 //std::cout<<"pdln="<<(l/2.0)*log(alpha_m/(N+alpha_m))<<"alpha_m="<<alpha_m<<std::endl;
 
 topGamma=gammalnmult(l,(N+alpha_w-n+l)/2.0);
 botGamma=gammalnmult(l,(alpha_w-n+l)/2.0);

 term2 = topGamma-botGamma-(l*N/2)*log(M_PI);


topdet = ((alpha_w-n+l)/2.0)*log(det(T(YY,YY)));
botdet = ((N+alpha_w-n+l)/2.0)*log(det(R(YY,YY)));
term3 = topdet-botdet; 

	return(term1+term2+term3);// the complete DAG score term
}

double envDAG::gammalnmult(const double l, const double xhalf){
 
 double x,myfactor,prod1;
 unsigned int j;
 x=2.0*xhalf; // conver to function of x not x/2
 myfactor=(l*(l-1)/4.0)*log(M_PI);
 prod1=0.0; // initial value - identity for sum
 for(j=1;j<=l;j++){
   prod1=prod1+lgamma( (x+1-j)/2.0 );
 }
 ////res = myfactor + sum(gammaln( (x+1-[1:l])/2 ));


  return(myfactor+prod1);

}


