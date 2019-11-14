function m = numdags(n)

% compute number of dags using orders - includes duplicates
tot=1;
for i=n:-1:2
    curtotNode=0;
    for j=0:(i-1) % from 0 through n-1, e.g. 1 2 3 | 4 so 
    	curtotNode=curtotNode+nchoosek(i-1,j); % 3,0 3,1 3,2 3,3 - sum up all parent possibilities for current node
    end
    tot=tot*curtotNode; % get total combinations in the order, e.g. node1 has 5 combs node2 has 3 combs etc
end

% finally multiply the combination in one order by all possible orders

m=factorial(n)*tot;

end
