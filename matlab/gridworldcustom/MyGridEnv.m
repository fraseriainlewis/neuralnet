classdef MyGridEnv < rl.env.MATLABEnvironment
    %MYENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        rewardTerminal = 10;
        rewardSpecial = 5; %alpha_mu
        reward= -1; % number of data points
        terminalState = [5 5];
        gridmap=reshape(1:25,5,5);

    end
    
    properties
        % Initialize system state to cell=2 (e.g. 2,1) '
        State = [2];
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = MyGridEnv()
            % Initialize Observation settings
            %% this is the DAG - a matrix
            ObservationInfo = rlFiniteSetSpec([1:25]);
            ObservationInfo.Name = 'DAG';
            ObservationInfo.Description = 'current DAG, n x n';
            
            % Initialize Action settings   
            ActionInfo = rlFiniteSetSpec([1 2 3 4]);
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            %updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            
            curlocidx=this.State;
            [r,c,v]=find(this.gridmap==curlocidx);
            curloc=[r c];
            newloc=curloc;

            % 2. determine action and generate new move
            if isequal(curloc,[2 4]) && Action ==4
                specialMove=1;
                newloc=[4 4];
                %disp('special move')
            else specialMove=0;	
            end

            if ~specialMove
                if Action==1 % move left
                newloc(2)=newloc(2)-1; % reduce col index by 1
                elseif Action==2 % move right
                newloc(2)=newloc(2)+1; % increment col index by 1
                elseif Action==3 % move up
                newloc(1)=newloc(1)-1; % decrement row index by 1
                elseif Action==4 % move down
                newloc(1)=newloc(1)+1; % increment row index by 1
                end
            end

            if newloc(1)>5 || newloc(1)<1 || newloc(2)>5 || newloc(2)<1
                newloc=curloc; % a move off the grid and so reset to current location as no move
            end	
            if isequal(newloc,[3 3]) || isequal(newloc,[3 4]) || isequal(newloc,[3 5]) || isequal(newloc,[4 3])
                newloc=curloc; % a move into a banned cell
            end
            
            disp('new loc')
            disp(newloc)

            newlocidx=this.gridmap(newloc(1),newloc(2));% back to index
            % Transform state to observation - store in LoggedSignals and also NextObs
            
            Observation = newlocidx; % repacks dag and loc into row vector

            % Update system states
            this.State = Observation;
            
            
            if isequal(newloc,this.terminalState)
                IsDone=1; % terminate as have found best DAG
                %disp('terminating')
            else 
            IsDone=0; % not yet reached terminal state
            end

            this.IsDone = IsDone;

            % Get reward
            if IsDone
                Reward = this.rewardTerminal; % found terminal state 
            elseif specialMove
                    Reward = this.rewardSpecial;
            else Reward= this.reward;    
            end
            
            disp('reward')
            disp(Reward)


        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            loc=[2];
            InitialObservation = loc;
            this.State = InitialObservation;          
           
        end
    end

end