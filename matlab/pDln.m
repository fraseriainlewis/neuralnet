function res = pDln(N,n,l,alpha_u,alpha_w,T,R,YYrow,YYcol)
  
 term1 = (l/2)*log(alpha_u/(N+alpha_u));

 topGamma=gammalnmult(l,(N+alpha_w-n+l)/2);
 botGamma=gammalnmult(l,(alpha_w-n+l)/2);

 term2 = topGamma-botGamma-(l*N/2)*log(pi);


topdet = ((alpha_w-n+l)/2)*log(det(T(YYrow,YYcol)));
botdet = ((N+alpha_w-1+l)/2)*log(det(R(YYrow,YYcol)));
term3 = topdet-botdet; 

res = term1+term2+term3 % the complete DAG score term

end;