clear all;
 cd '/Users/fraser/myrepos/neuralnet/matlab/griddag'
if true
	run dag_setup.m
	run script_DAGtablen4.m 
end


env=DAGenv(allStates,allScores);% template file defining class



rng(1000); %0 or 1000
InitialObs = reset(env,1)

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

% initialize environment
% first episode - run until termination or fixed number of steps? The success of the algorithm is the average number of steps 
% over episodes from start until reaches success.
s=1;
periodTotalReward=0;
reset(env,s); % set to initial state
% take a greedy action and update V(for currentstate) 
bestValue= -realmax;
    		for a = 1:15 % for each action
    			reset(env,s);
    			[NextState,Reward,IsDone,LoggedSignals] = step(env,actionLookup{a});
    			curQ=Reward + discount*V(s);
    			if curQ > bestValue
    				bestValue = curQ;
    				V(s) =curQ; % update value function for just this state
    				greedyA = a; % store best action
    				greedyNextState = NextState;
    				greedyReward = Reward; 
    			end;	
             end

%  set next state as result of greedy action and update
reset(env,greedyNextState);
periodTotalReward = periodTotalReward + greedyReward;




