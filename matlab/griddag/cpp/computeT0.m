% compute the T0 - prior precision for a DAG using Shachter and Kenley in Heckerman
clear all;
%cd '/Users/fraser/myrepos/neuralnet/matlab/griddag/cpp'
%cd '/home/lewisfa/myrepos/neuralnet/matlab/griddag/cpp'
%cd '/files/myrepos/neuralnet/matlab/griddag/cpp'
% imaginary sample sizes
alpha_w = 30;
alpha_m = 30;


if false
 % the hard coded example from Heckermand 92
  n=3;% number of variables in DAG
  % prior variance
  nu  = ones(1,n);
  % prior unconditional means
  mu0 = [0.1,-0.3,0.2];
  % reg coefs for each node
  b=zeros(n,n);
  b(1,2)=0.0;
  b(1,3)=1; b(2,3)=1;
end;

% - EXAMPLE 1 ----------------------------------------------------------
n=10;
if (alpha_w< n)
  disp("----------- error!!! alpha_w is too small -------------")
end;
% prior variance - equal 1 for simplicity
nu  = ones(1,n);
% prior unconditional means - equal 0 for simplicity
mu0 = zeros(1,n);

% reg coefs - format is b(1,1)=0 by construction, as first node in order. b(1,2) is first coef (1) for second node (2), can only have one parent
%                                                  b(1,3), b(2,3) are first and second coefs for third node, can only have two parents etc
b=zeros(n,n);
% for a chain DAG, a1, a2<-a1, a3<-a2, ... a10<-a9 is simple as only last allowed coef in each node is non-zero
b(1,2)=1;% node 2
b(2,3)=1;% node 3
b(3,4)=1;% node 4...
b(4,5)=1;
b(5,6)=1;
b(6,7)=1;
b(7,8)=1;
b(8,9)=1;
b(9,10)=1;% node 10

% get the precision matrix for the MVN which is equivalent to Graph, invert to get covariance matrix
sigmainv=priorPrec(nu,b);

%writematrix(inv(sigmainv),'covarN20a.csv');
%writematrix(mu0,'meansN20a.csv');
csvwrite('covarN10.csv',inv(sigmainv));
csvwrite('meansN10.csv',mu0);

% - EXAMPLE 2 ----------------------------------------------------------
n=20;
if (alpha_w< n)
  disp("----------- error!!! alpha_w is too small -------------")
end;
% prior variance - equal 1 for simplicity
nu  = ones(1,n);
% prior unconditional means - equal 0 for simplicity
mu0 = zeros(1,n);

% reg coefs - format is b(1,1)=0 by construction, as first node in order. b(1,2) is first coef (1) for second node (2), can only have one parent
%                                                  b(1,3), b(2,3) are first and second coefs for third node, can only have two parents etc
b=zeros(n,n);
% [a1][a2|a1][a3][a4|a1][a5][a6|a4:a3][a7|a6][a8][a9][a10|a4:a6:a7][a11][a12][a13|a11:a12][a14][a15:a2][a16][a17][a18|a10][a19][a20|a11:a19]
b(1,2)=1;% node 2
b(1,4)=1;% node 4...
b(3,6)=1;b(4,6)=1;
b(6,7)=1;
b(4,10)=1;b(6,10)=1;b(7,10)=1;% node 10
b(11,13)=1;b(12,13)=1;
b(2,15)=1;
b(10,18)=1;
b(11,20)=1;b(19,20)=1;

% get the precision matrix for the MVN which is equivalent to Graph, invert to get covariance matrix
sigmainv=priorPrec(nu,b);

%writematrix(inv(sigmainv),'covarN20a.csv');
%writematrix(mu0,'meansN20a.csv');
csvwrite('covarN20a.csv',inv(sigmainv));
csvwrite('meansN20a.csv',mu0);


% Compute T = precision matrix in Wishart prior.
%%%%%%% equation 5 and 6 in Geiger and Heckerman
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

T

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% - EXAMPLE 3 ----------------------------------------------------------

n=30;
if (alpha_w< n)
  disp("----------- error!!! alpha_w is too small -------------")
end;
% prior variance - equal 1 for simplicity
nu  = ones(1,n);
nu(1,20)=100;
nu(1,19)=10;
nu(1,3)=50;
% prior unconditional means - equal 0 for simplicity
mu0 = zeros(1,n);

%mu0(1,10:25)=1.5;
%% reg coefs - format is b(1,1)=0 by construction, as first node in order. b(1,2) is first coef (1) for second node (2), can only have one parent
%%                                                  b(1,3), b(2,3) are first and second coefs for third node, can only have two parents etc
b=zeros(n,n);
%% [a1][a2][a3|a2][a4][a5|a4][a6][a7][a8][a9][a10][a11|a9][a12|a6][a13|a5:a6][a14][a15][a16|a13][a17]
%% [a18][a19|a14:a17][a20|a3:a16]
b(3,20)=1;b(16,20)=1;%% node 20...
b(14,19)=1;b(17,19)=1;%% node 19...
b(13,16)=1;%% node 16...
b(5,13)=1;b(6,13)=1;%% node 13
b(6,12)=1;
b(9,11)=1;
b(4,5)=1;
b(2,3)=1;
b(25,30)=1;
b(1,30)=1;
b(13,30)=1;
% get the precision matrix for the MVN which is equivalent to Graph, invert to get covariance matrix
sigmainv=priorPrec(nu,b);

%writematrix(inv(sigmainv),'covarN20b.csv');
%writematrix(mu0,'meansN20b.csv');
csvwrite('covarN30b.csv',inv(sigmainv))
csvwrite('meansN30b.csv',mu0)


% Compute T = precision matrix in Wishart prior.
%%%%%%% equation 5 and 6 in Geiger and Heckerman
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

T
