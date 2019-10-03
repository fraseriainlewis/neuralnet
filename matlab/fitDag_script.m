%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% above this is all necessary for any network score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log(DAG)= sum over each node
%           for each node there are two terms A-B, A=pDln(...,[node and parents])
%                                                  B=pDln(...,[node only])
% so for each node we need it's parents

run dag_setup.m;

% note - need to check cycle

dag=[0 0 1;
     1 0 0;
     0 1 0];

fitDAG(dag,N,alpha_m,alpha_w,T,R)

dag=[0 0 1;
     1 0 1;
     0 1 0];

fitDAG(dag,N,alpha_m,alpha_w,T,R)

dag=[0 0 0;
     1 0 0;
     0 0 0];     

fitDAG(dag,N,alpha_m,alpha_w,T,R)

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
