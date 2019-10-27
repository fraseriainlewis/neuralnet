clear all;
mu0 = [0,0,0];
alpha_w = 6;
alpha_u = 6;
n=3;
T0scalar=alpha_u*(alpha_w-n-1)/(alpha_u+1);
T0 = diag([T0scalar T0scalar T0scalar],0);
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

TN = T0 + cov(thedata)*(N-1)+(alpha_w*N)/(alpha_w+N)*(mu0-mean(thedata)).*(mu0-mean(thedata))'

C = (-N/2)*log(pi) + (1/2)*log(alpha_u/(alpha_u+N));

score1 = C - gammaln( (alpha_w-n+1)/2) + gammaln((alpha_w+N-n+1)/2) +((alpha_w-n+1)/2)*log(T0scalar) - ((alpha_w+N-n+1)/2)*log(TN(1,1))

