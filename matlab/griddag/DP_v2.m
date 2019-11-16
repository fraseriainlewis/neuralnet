clear all;
 cd '/home/lewisfa/myrepos/neuralnet/matlab/griddag'
if true
	run dag_setup.m
	run script_DAGtablen4.m 
end


env=DAGenv(allStates,allScores);% template file defining class



rng(1000); %0 or 1000
InitialObs = reset(env,1,0)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,[2 1]);

[r,c]=size(allStates);% 17 row 8688 cols
numStates=c;
% create vector to hold value for each state
% initialize to zero
V=zeros(c,1); % 8688 by 1
terminal=find(allScores>=max(allScores)); % indexes of the best score - terminal condition to value function is zero
policy=randi([1 15],c,1); % each row is a state the entry is the action form the policy - 1 through 15
policyStable=zeros(c,1);
discount=0.90;


actionLookup={[0 1],[0 0],[0 -1],... % no spatial move 
                              [1 1],[1 0],[1 -1],...             % left
                              [2 1],[2 0],[2 -1],...             % right
                              [3 1],[3 0],[3 -1],...             % up
                              [4 1],[4 0],[4 -1]};
delete 'mylog.txt'
diary on
diary 'mylog.txt'

for i=1:50  % this is the number of times to do policy iteration - perfect fit is all but 
disp('------------ outer iter ----------')
disp(i)
delta=1;% large than tolerance
while delta>0.01 
	delta=0;
	for s = 1:numStates
		%disp('iter')
		%disp(s)
    	% get current value for current state
    	if ~ismember(s,terminal)  % terminal state set to zero by construction
    		curV = V(s);
    		% set to state s 
    		reset(env,s,0);
    		[NextState,Reward,IsDone,LoggedSignals] = step(env,actionLookup{policy(s)});% get reward from current policy-action
    		% now do update 
        	V(s) = Reward + discount*V(NextState);
        	delta=max([delta,abs(curV-V(s))]);
    	end % end if  
	end % end for
   %disp('delta')
   %disp(delta)
end % end while  	

disp('end of value est')

% now update policy
for s = 1:numStates
	%disp('policy eval')
	%disp(s)
	curP = policy(s);
	% set to state s and find best action of all possible
    		reset(env,s,0);
    		bestValue= -realmax;
    		for a = 1:15
    			[NextState,Reward,IsDone,LoggedSignals] = step(env,actionLookup{a});
    			curQ=Reward + discount*V(NextState);
    			if curQ > bestValue
    				bestValue = curQ;
    				policy(s) = a; %update policy to current action which is new best
    				% TO-DO what about breaking ties? 
    			elseif curQ==bestValue
    				% randomly break ties
    				if rand>=0.5
    				bestValue = curQ; % not actually needed
    				policy(s) = a; %update policy to current action which is new best
    				end
    			end;	
             end

    if policy(s) == curP % so policy has not changed
       policyStable(s)=1; % mark policy as stable/converged for this state
    end

end
disp('end of policy update')
disp('how many are stable')
[r,c]=size(find(policyStable~=0))
disp(r)
%disp(histcounts(policy))

end

diary off


% check the policy
% check pick a state in AllState, e.g. 3995 which does not have the target network score and then see if the 
% policy for this state when taken then gives a DAG this is in the target state (assuming it can reach target from just one action)
%mydag=reshape(allStates(1:16,1,4,4)';
%fitDAG(mydag,N,alpha_m,alpha_w,T,R)

save 'DPworkspace2.mat'

exit
