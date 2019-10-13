clear all;
disp("SETUP FOR FITTING DAGS - loads in data set and some pre-computation. Dimension is fixed")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a setup script necessary for running before doing a structure search across DAGS
% SET parameters: filename, alpha_w, alpha_m, and priorType
% if priorType=2 then need to additional manually set nu, mu0, b[]
%
% It should be run before the DAG structure search:
% It sets up: 
% 1. the observed data - set the file name below
% 2. the priors - set alpha_w, alpha_m, nu, mu0, b[]
% 3. computations needed to compute log Scores, irrespective of the specific DAG structure  
%
% notes: 1. the prior used by default is a very simple independence DAG, precision = 1 for all nodes and zero means.
%           this is the same prior used by CRAN BiDAG, and so enables QC check with scores from BiDAG. The far more
%           elaborate prior - giving a prior DAG with regression coefficients as per Geiger and Heckerman 1994 is 
%           implemented (although the checks are not so extensive - just that it matches the example T0 in Geiger
%           so use with caution)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% csv file with data - no header
filename='n4m1000.csv'; %'data_matrix.txt';
disp("NOTE: CURRENT DATA SET HAS FIVE VARIABLES")
% imaginary sample sizes
alpha_w = 30;
alpha_m = 30;
% type of prior - default indep or something more elaborate
priorType=1;

% raw data
thedata = readmatrix(filename,'Delimiter',','); % the data
[nrow,ncol]=size(thedata); % get number of obs N, and number of variables n
N=nrow;
n=ncol;

if priorType==1 % simple BiDAG prior
  nu  = ones(1,n);  % prior precision for each node = 1
  mu0 = zeros(1,n); % prior mu0 for each node = 0
  b=zeros(n,n);     % this is the regression coefs - all zero 

elseif priorType==2 % Geiger and Heckerman 1994 prior with prior DAG - example in manuscript
	disp("WARNING: using Geiger and Heckerman prior DAG using Shachter and Kenley method - are you sure?")
	nu  = ones(1,n);
    mu0 = [0.1,-0.3,0.2];
    b=zeros(n,n); 
    b(1,2)=0.0;               
    b(1,3)=1; b(2,3)=1;
else error('priorType is not a 1 or 2!!')    
end

% Compute T = precision matrix in Wishart prior.
%%%%%%% equation 5 and 6 in Geiger and Heckerman
sigmainv=priorPrec(nu,b);
%
% We need precision matrix T which defines the prior Wishart distribution. 
% Basic method: Equation 20 in 2002 defines the covariance of X as a function of (T^prime)^-1 we know the cov of X, it's just inv(sigmainv) from Equation 5. 
% so we have the left hand side of Equation 20. Now equation 19 gives us an expression for T^prime = RHS, and so (T^prime)^-1 is just the inverse of the RHS of equation 19
% which reduces to inv(sigmainv) = (alpha_w-n+1)/(alpha_w-n-1)*(alpha_m+1)/(alpha_m*(alpha_w-n_1)) * T, and cancelling the terms and 
% re-arranging gives T = inv(sigmainv)/sigmaFactor as below. Lots of faff but easy enough.   
sigmaFactor = (alpha_m+1)/(alpha_m*(alpha_w-n-1));
% T = precision matrix T for use in wishart prior")
T=inv(sigmainv)/sigmaFactor;
%% this matches the T0 matrix values given in 1994 Heckerman - so works ok. 

% Compute R = precision matrix term in score expression
% note using equation A.15 in Kuipers et al 2014 in the SI which is same as from Heckerman 1994 (eqn 9) and Heckerman 2003 (eqn 17) - note as per email with Kuipers in Sept
% the equation 4 in Kuipers main manuscript incorrecly has alpha_w rather than alpha_m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xbarL=mean(thedata);
R = T + cov(thedata)*(N-1) + (alpha_m*N)/(alpha_m+N) * (mu0-xbarL).*(mu0-xbarL)';
% this matches matrix values given in 1994 Heckerman - but this because alpha_w=alpha_m in that example. 




