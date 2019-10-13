function [envConstants] = setEnvConstants()


% now store in struct
% imaginary sample sizes
envConstants.rewardTerminal = 10;
envConstants.rewardSpecial = 5; %alpha_mu
envConstants.reward= -1; % number of data points
envConstants.terminalState = [5 5];
envConstants.map=reshape(1:25,5,5);

end

