function res = gammalnmult(l,xhalf)
 x=2*xhalf; % conver to function of x not x/2
 myfactor=(l*(l-1)/4)*log(pi);
 %prod1=0.0; % initial value - identity for sum
 %for j=1:l
 %  prod1=prod1+gammaln( (x+1-j)/2 );
 %end;
 res = myfactor + sum(gammaln( (x+1-[1:l])/2 ));
 %res= myfactor+prod1; 
end  