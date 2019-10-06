%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% above this is all necessary for any network score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log(DAG)= sum over each node
%           for each node there are two terms A-B, A=pDln(...,[node and parents])
%                                                  B=pDln(...,[node only])
% so for each node we need it's parents

run dag_setup.m;

% note - need to check cycle

dag=[0 0 0;
     0 0 0;
     0 0 0];


dag1=[0 0 1;
     1 0 0;
     0 1 0];

dag2=[0 0 1;
     1 0 1;
     0 1 0];

dag3=[0 0 0;
     1 0 0;
     0 0 0];     

fitDAG(dag0,N,alpha_m,alpha_w,T,R)
fitDAG(dag1,N,alpha_m,alpha_w,T,R)
fitDAG(dag2,N,alpha_m,alpha_w,T,R)
fitDAG(dag3,N,alpha_m,alpha_w,T,R)

[n,m]=size(dag0);
tmpDAG=zeros(n,n,'uint32');
tmpVec1=zeros(1,n,'uint32');
tmpVec2=zeros(1,n,'uint32');
tmpVec3=zeros(1,n,'uint32');

dag0=uint32(dag0);
hasCycle=cycle(dag0,tmpDAG,tmpVec1,tmpVec2,tmpVec3);

dag4=dag0;
dag4(3,1)=1;
dag4(1,3)=1;
hasCycle=cycle(dag4,tmpDAG,tmpVec1,tmpVec2,tmpVec3);









%{
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
%}
