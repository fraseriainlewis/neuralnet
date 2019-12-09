function res = pDln(N,n,l,alpha_m,alpha_w,T,R,YY)
  
 term1 = (l/2)*log(alpha_m/(N+alpha_m));

 topGamma=gammalnmult(l,(N+alpha_w-n+l)/2);
 botGamma=gammalnmult(l,(alpha_w-n+l)/2);

 term2 = topGamma-botGamma-(l*N/2)*log(pi);


topdet = ((alpha_w-n+l)/2)*log(det(T(YY,YY)));
botdet = ((N+alpha_w-n+l)/2)*log(det(R(YY,YY)));
term3 = topdet-botdet; 

%disp('plnd')
%disp((l/2)*log(alpha_m/(N+alpha_m)))
%disp(log(alpha_m/(N+alpha_m)))
%disp((l/2))

res = term1+term2+term3; % the complete DAG score term

end