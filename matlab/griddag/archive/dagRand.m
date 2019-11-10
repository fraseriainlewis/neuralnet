function m = dagrng(n)

% generate random DAG
%1. generate random permutation of 1...n
%e.g. 2 1 3 5 4
%2. then randomly choose parents for each child node following the order

m=zeros(n,n);
ord=randperm(n); % e.g. 2 3 1 4

for i=2:n
  % start i=2 since i=1 has no parents by construction
  child=ord(i);% child node id e.g. 3
  potPars=ord(1:(i-1)); % parent nodes (e.g. before 3 in sequence e.g. 2 )
  % choose each parent in potParts with 0.5 prob
  paridx=find(rand(1,i-1)>0.5); % this gives 0 1 0 0 1 etc., 1= make parent, 0 = not
                                      % then chooses the id which are =1
  actPars=potPars(paridx);  % this is the ids which are chosen to be parents of child id (child)
  m(child,actPars)=1; % update dag - add to row child the cols which are parents

end


end
