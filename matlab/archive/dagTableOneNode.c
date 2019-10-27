/** /Applications/MATLAB_R2019a.app/bin/mex -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas arrayProductA.c -R2018a
  **/
/** call this from matlab prompt
 cb=(nchoosek(3,0)+nchoosek(3,1)+nchoosek(3,2)+nchoosek(3,3))
  a=dagTableSize(zeros(3,3,'uint32'),cb);
 b=combvec(a',a');
 dagstoreflat=combvec(a',b);
 i=12;
 reshape(dagstoreflat(:,i),3,3)' % e.g. a single DAG
*/
/*==========================================================
 * arrayProduct.c - example in MATLAB External Interfaces
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs a 1xN matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = arrayProduct(multiplier, inMatrix)
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2007-2012 The MathWorks, Inc.
 *
 *========================================================*/

#include "mex.h"
#include "math.h"
#include <gsl/gsl_combination.h>
/* The computational routine */
/* this computes n choose r, for n=numnodes and r from 0 to numnodes, so gives total number of 
 * possible parent combinations per node. To get total DAGs (ignoring cycles) this is just the
 * paraent combination raised to power of numnodes */
void dagTableOneNode(unsigned int *dag, double totalComb,double *tablesize, mwSize numnodes)
{
    mwSize i,j,k;
    gsl_combination *c;
    mwSize index=0;
    mwSize nrows=(mwSize)totalComb;
    mwSize curparentindex;
    /*mexPrintf ("All subsets by size:\n") ;*/
  for (i = 0; i <= numnodes; i++)
    { /** for each cardinality of parents */
      c = gsl_combination_calloc (numnodes, i);
      do
        { /** generate all combinations */
          /** take current combination and roll into a row in the results array */
          for(j=0;j<=numnodes;j++){tablesize[index+j*nrows]=0.0;}/** reset the node parents to independence **/
          for(k=0;k<gsl_combination_k(c);k++){curparentindex=gsl_combination_get (c, k);
                                              tablesize[index+curparentindex*nrows]=1.0;}
          
          index++;  
      }
      while (gsl_combination_next (c) == GSL_SUCCESS);
      gsl_combination_free (c);
    }

    
    mexPrintf("here %u\n",index);
    //tablesize[0]=pow(total,numnodes);
    
}


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned int *inDAGempty;               /* 1xN input matrix */
    double inTotalComb;
    size_t nnodes;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","2 inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }

    /* make sure the first input argument is type uint32 */
    if( ! (mxIsUint32(prhs[0]))) 
         {
        mexErrMsgIdAndTxt("MyToolbox:cycle:notUint32","All inputs must be type uint32.");
    }
    
    /* check that number of rows in first input argument is 1 */
    if(mxGetM(prhs[0])!=mxGetN(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a square matrix.");
    }
    
    /* get the value of the scalar input  */
    inTotalComb = mxGetScalar(prhs[1]);
    
    /* create a pointer to the real data in the input matrix  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    inDAGempty = mxGetUint32s(prhs[0]);
    #else
    inMatrix = mxGetPr(prhs[0]);
    #endif

    /* get dimensions of the input matrix */
    nnodes = mxGetM(prhs[0]);

    /* create the output matrix - using 1x1 - could use scalar instead */
    plhs[0] = mxCreateDoubleMatrix((mwSize)inTotalComb,(mwSize)nnodes,mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    outMatrix = mxGetDoubles(plhs[0]);
    #else
    outMatrix = mxGetPr(plhs[0]);
    #endif

    /* call the computational routine */
    dagTableOneNode(inDAGempty,inTotalComb,
             outMatrix,(mwSize)nnodes);
}




