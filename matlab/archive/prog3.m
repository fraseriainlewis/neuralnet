clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Compute network scores as per example 3.5 in Geiger and Heckerman 1994
%
% note - this method is quite complicated - but give all the nice properties and conjugacy, but fiddly. 
% note1: it seems the computation of the constant c(n,alpha) in the example is not correct, but it's possible to replicate T0 and T20
% note2: to compute T0 we use Geiger and Heckerman 2003 - eqn 19 and 20 (similar to 18 and 19 in 1994 but different notation)
% note3: we use the network score from Kuipers, Moffa and Heckerman 2014 - slightly different formula (less bias)
%
% General method. 
% 1. Setup we have variables x1,x2,x3 and we have a prior DAG (see Figure 1. in 1994 article) and mean vector (marginal prior mean for each node), 
%    mu0=(0.1, -0.3, 0.2)
%    nu0=(1,1,1) % prior precision in each node
%    b2 = (0) % line vector for x2 regression coefficients
%    b3 = (1,1) % line vector for x3 regression coefficients
%    alpha_w = 6 hyper parameter for precision in normal-Wishart (u,W) dist alpha_w>n+1, n=dimension=3 
%    alpha_m = 6 hyper paramters for mean in normal-Wishart (u,W)
%
% 2. Compute T = precision matrix in Wishart prior. See eqn 5 and 6 in Heckerman 1994, and then eqn 19 and 20 in Heckerman 2003 
%    (or eqn 18 + 19 in 1994 but slightly different notation)
%
% 3. 2. Compute R in eqution 2 in Kuipers, Moffa and Heckerman - when put together with equation 1 in Kuipers this is the network score
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. SETUP - use data in 1994 example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=3; % dimension of total DAG - total number of variables
nu  = [1,1,1];
%mu0 = [0.1,-0.3,0.2];
mu0 = [0,0,0];
%b2  = [0];
%b3  = [1;1];
% setup b col vectors as matrix
b=zeros(n,n); %storage - note we will never used top row or first col
              % as b1 is not defined - each DAG must have one node without a parent
%b(1,2)=0.0;               
%b(1,3)=1; b(2,3)=1;

alpha_w = 6;
alpha_m = 6;

N=20; % number of observations
thedata=[-0.78 -1.55  0.11;
          0.18 -3.04 -2.35;
          1.87  1.04  0.48;
         -0.42  0.27 -0.68;
          1.23  1.52  0.31;
          0.51 -0.22 -0.60;
          0.44 -0.18  0.13;
          0.57 -1.82 -2.76;
          0.64  0.47  0.74;
          1.05  0.15  0.20;
          0.43  2.13  0.63;
          0.16 -0.94 -1.96;
          1.64  1.25  1.03;
         -0.52 -2.18 -2.31;
         -0.37 -1.30 -0.70;
          1.35  0.87  0.23;
          1.44 -0.83 -1.61;
         -0.55 -1.33 -1.67;
          0.79 -0.62 -2.00;
          0.53 -0.93 -2.92;
          ];

%writematrix(thedata,"data.txt");
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Compute T = precision matrix in Wishart prior.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% equation 5 and 6 in Geiger - this is manual and neededs automated
disp("sigmainv")
sigmainv=priorPrec(nu,b)

%%%%%%%%%%%%%%%%%%%%
% We need precision matrix T which defines the prior Wishart distribution. 
% Basic method: Equation 20 in 2003 defines the covariance of X as a function of (T^prime)^-1 we know the cov of X, it's just inv(sigmainv) from Equation 5. 
% so we have the left hand side of Equation 20. Now equation 19 gives us an expression for T^prime = RHS, and so (T^prime)^-1 is just the inverse of the RHS of equation 19
% which reduces to inv(sigmainv) = (alpha_w-n+1)/(alpha_w-n-1)*(alpha_m+1)/(alpha_m*(alpha_w-n_1)) * T, and cancelling the terms gives 
% so re-arranging gives T = inv(sigmainv)/sigmaFactor as below. Lots of faff but easy enough.   
sigmaFactor = (alpha_m+1)/(alpha_m*(alpha_w-n-1));
disp("This is precision matrix T for use in wishart prior")
T=inv(sigmainv)/sigmaFactor
%% this matches the T0 matrix values given in 1994 Heckerman - so works ok. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Compute R = precision matrix term in score expression
% note using equation 4 in Kuipers et al 2014 which is different from Heckerman 1994 (eqn 9) and Heckerman 2003 (eqn 17) as the latter two use alpha_m where in Kuipers
% they use alpha_w - suspect Kuipers is correct. If they are same it matter not, i.e. alpha_w=alpha_m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xbarL=mean(thedata);

disp("new!!")
R = T + cov(thedata)*(N-1) + (alpha_m*N)/(alpha_m+N) * (mu0-xbarL).*(mu0-xbarL)'
% this matches matrix values given in 1994 Heckerman - but this because alpha_w=alpha_m in that example. 




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% above this is all necessary for any network score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log(DAG)= sum over each node
%           for each node there are two terms A-B, A=pDln(...,[node and parents])
%                                                  B=pDln(...,[node only])
% so for each node we need it's parents
dag=[0 0 1;
     1 0 0;
     0 1 0];
[nrow,ncol]=size(dag); 
totLogScore=0.0;    
for i=1:nrow
    % process node i
    nodei=dag(i,:);
    par_idx=find(nodei);% indexes of parents of node i
    [tmp,npars]=size(par_idx);
    if npars==0 
      disp("no parents")
      % we are done as p(d) = singleX/1.0
      YY=[i];
      [tmp,l] = size(YY); % l=dimension of d
      totLogScore=totLogScore+pDln(N,n,l,alpha_m,alpha_w,T,R,YY);
    else disp("have parents")
      % if we have parents then we need to compute A/B, A = parents U node, B = parents
      YY=[par_idx i];
      [tmp,l] = size(YY); % l=dimension of d
      A=pDln(N,n,l,alpha_m,alpha_w,T,R,YY);
      YY=[par_idx];
      [tmp,l] = size(YY); % l=dimension of d
      B=pDln(N,n,l,alpha_m,alpha_w,T,R,YY);
      totLogScore=totLogScore+A-B;
    end  
end 
sprintf("log score for DAG %f\n", totLogScore )

%{
%% now try network x1 x2->x3
l=1;
YYrow=[1];
YYcol=[1];
disp("This is log P(d|X1) ")
logp_dX1 =pDln(N,n,l,alpha_m,alpha_w,T,R,YYrow) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c. p_X2  
l=1; % two variables
YYrow=[2];
YYcol=[2];
disp("This is log P(d|X2) ")
logp_dX2 =pDln(N,n,l,alpha_m,alpha_w,T,R,YYrow) % the complete DAG score term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% c. p_X3  
l=1; % two variables
YYrow=[3];
YYcol=[3];
disp("This is log P(d|X3) ")
logp_dX3 =pDln(N,n,l,alpha_m,alpha_w,T,R,YYrow) % the complete DAG score term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp("This is ln score x1 x2 x3")
logp_dX1+logp_dX2+logp_dX3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% b. p_dX3X2  
l=2; % two variables
YYrow=[2 3];
YYcol=[2 3];
disp("This is log P(d|X2,X3) ")
logp_dX2X3 =pDln(N,n,l,alpha_m,alpha_w,T,R,YYrow) % the complete DAG score term
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp("This is ln score x1 x2->x3")
score=logp_dX1+logp_dX2X3

disp("This is ln score x1 x2->x3: longhand")
score=logp_dX1+logp_dX2+logp_dX2X3-logp_dX2
%}

















