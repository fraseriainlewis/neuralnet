function [envConstants] = dagEnvConstants(datafilename,alpha_w,alpha_m)

% This is a setup script necessary for running before doing a structure search across DAGS
% SET parameters: datafilename, alpha_w, alpha_m, note we assume the simple prior from BiDAG
% which is simpler than Heckerman
%
% It should be run before the DAG structure search:
% It sets up: 
% 1. the observed data - set the file name below
% 2. the priors - set alpha_w, alpha_m, nu, mu0, b[]
% 3. computations needed to compute log Scores, irrespective of the specific DAG structure 

% notes: 1. the prior used by default is a very simple independence DAG, precision = 1 for all nodes and zero means.
%           this is the same prior used by CRAN BiDAG, and so enables QC check with scores from BiDAG. 

% read raw data - assume CSV in current working directory
thedata = readmatrix(datafilename,'Delimiter',','); % the data
[nrow,ncol]=size(thedata); % get number of obs N, and number of variables n
N=nrow;% number of data records in data
n=ncol;% number of nodes in DAG - number of variables in data


nu  = ones(1,n);  % prior precision for each node = 1
mu0 = zeros(1,n); % prior mu0 for each node = 0
b=zeros(n,n);     % this is the regression coefs - all zero 

% Compute T = precision matrix in Wishart prior.
%%%%%%% equation 5 and 6 in Geiger and Heckerman
sigmainv=priorPrec(nu,b);
%
% We need precision matrix T which defines the prior Wishart distribution. 
% Basic method: Equation 20 in Geiger 2002 defines the covariance of X as a function of (T^prime)^-1 we know the cov of X, it's just inv(sigmainv) from Equation 5. 
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

% create scratch storage for cycle checks 
tmpDAG=zeros(n,n,'uint32');
tmpVec1=zeros(1,n,'uint32');
tmpVec2=zeros(1,n,'uint32');
tmpVec3=zeros(1,n,'uint32');

% now store in struct
% imaginary sample sizes
envConstants.alpha_w = alpha_w;
envConstants.alpha_m = alpha_m; %alpha_mu
envConstants.N=N; % number of data points
envConstants.n=n; % number of nodes/variables
envConstants.T=T; % needed for score calc
envConstants.R=R; % needed for score calc

envConstants.tmpDAG=tmpDAG;
envConstants.tmpVec1=tmpVec1;
envConstants.tmpVec2=tmpVec2;
envConstants.tmpVec3=tmpVec3;

end

