function res = gammamult(l,xhalf)
 x=2*xhalf; % conver to function of x not x/2
 myfactor=pi^(l*(l-1)/4);
 prod1=1.0; % initial value - identity for product
 for j=1:l
   prod1=prod1*gamma( (x+1-j)/2 );
 end;
 res= myfactor*prod1; 
end  