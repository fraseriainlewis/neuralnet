%% setup environment
ObservationInfo = rlNumericSpec([18 1]);
ObservationInfo.Name = 'CartPole States';
ObservationInfo.Description = 'x, dx, theta, dtheta';

ActionInfo = rlFiniteSetSpec([-10 10]);
ActionInfo.Name = 'CartPole Action';


% Acceleration due to gravity in m/s^2
envConstants.Gravity = 9.8;
% Mass of the cart
envConstants.MassCart = 1.0;
% Mass of the pole
envConstants.MassPole = 0.1;
% Half the length of the pole
envConstants.Length = 0.5;
% Max Force the input can apply
envConstants.MaxForce = 10;
% Sample time
envConstants.Ts = 0.02;
% Angle at which to fail the episode
envConstants.ThetaThresholdRadians = 12 * pi/180;
% Distance at which to fail the episode
envConstants.XThreshold = 2.4;
% Reward each time step the cart-pole is balanced
envConstants.RewardForNotFalling = 1;
% Penalty when the cart-pole fails to balance
envConstants.PenaltyForFalling = -5;

StepHandle = @(Action,LoggedSignals) myStepFunction2(Action,LoggedSignals,envConstants);
ResetHandle = @myResetFunction;

env2 = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);


%env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');


rng(0);
InitialObs = reset(env2)

[NextObs,Reward,IsDone,LoggedSignals] = step(env2,10);
