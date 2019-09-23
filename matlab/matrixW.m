
nu  = [1,1.,1.];
b2  = [0.0];
b3  = [-1.3;1.5];
n=3;

% setup b col vectors as matrix
b=zeros(n,n); %storage - note we will never used top row or first col
              % as b1 is not defined - each DAG must have one node without a parent
b(1,2)=0.0;               
b(1,3)=-1.3; b(2,3)=1.5;
%b
%kenley1(b,2)
%kenley1(b,3)

%b2=column 2 down to above diagonal
%b3=column 3 down to above diagonal

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Compute T = precision matrix in Wishart prior.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% equation 5 and 6 in Geiger - this is manual and neededs automated
wsize=1;
w1  = zeros(wsize,wsize);
w1(1,1) = 1/nu(1);
w1

wsize=2;
w2  = zeros(wsize,wsize); % initialize
w2(1:(wsize-1),1:(wsize-1)) = w1+kron(e1(b,2),e1(b,2)')/nu(2); % set submatrix top corner
w2(wsize,1:(wsize-1))= -e1(b,2)'/nu(2); % last row
w3(1:(wsize-1),wsize)= -e1(b,2)/nu(2); % last col
w2(wsize,wsize) = 1/nu(2); % bottom right cell
w2

wsize=3;
w3  = zeros(wsize,wsize); % initialize
w3(1:(wsize-1),1:(wsize-1)) = w2+kron(e1(b,3),e1(b,3)')/nu(3); % set submatrix top corner
w3(wsize,1:(wsize-1))= -e1(b,3)'/nu(3); % last row
w3(1:(wsize-1),wsize)= -e1(b,3)/nu(3); % last col
w3(wsize,wsize) = 1/nu(3); % bottom right cell
w3

w3new=priorPrec(nu,b)

% check - this uses long hand formula in equation 6 as a check - only works for b2=0!!
wcheck=zeros(wsize,wsize);
wcheck(1,1)=1/nu(1)+(b3(1)^2)/nu(3);
wcheck(1,2)=(b3(1)*b3(2))/nu(3);
wcheck(1,3)=-b3(1)/nu(3);
wcheck(2,2)=(1/nu(2))+(b3(2)^2)/nu(3);
wcheck(2,3)=-b3(2)/nu(3);
wcheck(3,3)=1/nu(3);

wcheck(2,1)=wcheck(1,2);
wcheck(3,1)=wcheck(1,3);
wcheck(3,2)=wcheck(2,3);

wcheck
% check that code computation is same as manual long hand
if (isequal(w3,wcheck))
	disp("correct match - precision matrix of prior DAG");
	sigmainv=w3;
else disp("error!");
	 clear sigmainv;
end;