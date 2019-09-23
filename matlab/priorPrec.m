function m = priorPrec(nu,b)
 
 n=size(b)(1); % number of variables

 wsize=1;
 w1  = zeros(wsize,wsize);
 w1(1,1) = 1/nu(1);
 % w1 is current matrix
 % w2 is new matrix

 for i=2:n
  wsize=i;
  w2  = zeros(wsize,wsize); % initialize
  w2(1:(wsize-1),1:(wsize-1)) = w1+kron(e1(b,i),e1(b,i)')/nu(i); % set submatrix top corner
  w2(wsize,1:(wsize-1))= -e1(b,i)'/nu(i); % last row
  w2(1:(wsize-1),wsize)= -e1(b,i)/nu(i); % last col
  w2(wsize,wsize) = 1/nu(i); % bottom right cell

  w1=w2; % update copy new matrix to current matrix
 end;
 
 m = w2; % return the final matrix Wn

end;