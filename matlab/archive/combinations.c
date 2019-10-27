/** call this from matlab prompt
 a=cycle(uint32(b),zeros(3,3,'uint32'),ones(1,3,'uint32'),zeros(1,3,'uint32'),zeros(1,3,'uint32'))
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
void get_numincomingedges(unsigned int *incomingedges,unsigned int *graph, unsigned int numnodes);

#include <gsl/gsl_combination.h>
/* The computational routine */
void dagTableSize(unsigned int *dag, mwSize numnodes)
{
    
    mexPrintf("here");
    
}


/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned int *inDAGempty;               /* 1xN input matrix */
    size_t nnodes;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","1 input required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }

    /* make sure the first input argument is type uint32 */
    if( ! (mxIsUint32(prhs[0]))) 
         {
        mexErrMsgIdAndTxt("MyToolbox:cycle:notUint32","All inputs must be type uint32.");
    }
    
    /* check that number of rows in second input argument is 1 */
    if(mxGetM(prhs[0])!=mxGetN(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a square matrix.");
    }
    
    /* create a pointer to the real data in the input matrix  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    inDAGempty = mxGetUint32s(prhs[0]);
    #else
    inMatrix = mxGetPr(prhs[0]);
    #endif

    /* get dimensions of the input matrix */
    nnodes = mxGetM(prhs[0]);

    /* create the output matrix - using 1x1 - could use scalar instead */
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    outMatrix = mxGetDoubles(plhs[0]);
    #else
    outMatrix = mxGetPr(plhs[0]);
    #endif

    /* call the computational routine */
    dagTableSize(inDAGempty,
             outMatrix,(mwSize)nnodes);
}




