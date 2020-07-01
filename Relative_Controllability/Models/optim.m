function r = optim(r, prc_fun, obs_fun, opt_algo)

% Use means of priors as starting values for optimization for optimized parameters (and as values
% for fixed parameters)
init = [r.c_prc.priormus, r.c_obs.priormus]; % 모든 prior parameter 의 mu(log sa 도 포함 sigma>0 이어야 하므로), 
%rho (뭔지 모르지만 현재 필요 없음), kappa, omega
% level = 3 로 default 이며 omega 의 level 1 은 NaN, 3 은 theta 에 해당한다, 또한 모든
% 경우에서 level 1은 deterministic 하므로 (level 2에 의해 결정), mean/std 가 NaN 으로 정의

% Determine indices of parameters to optimize (i.e., those that are not fixed or NaN)
% 어떤 parameter 를 free 로 둘 것인가? 기본적으로는 omega (2) 및 zeta(response model) 만 free로 두는듯
opt_idx = [r.c_prc.priorsas, r.c_obs.priorsas];
opt_idx(isnan(opt_idx)) = 0;
opt_idx = find(opt_idx); % free parameter 의 index

% Number of perceptual and observation parameters 
% perceptual, observational model 각각의 파라미터 갯수
n_prcpars = length(r.c_prc.priormus);
n_obspars = length(r.c_obs.priormus);

% Construct the objective function to be MINIMIZED:
% The negative log-joint as a function of a single parameter vector
% Objective function 을 정의하고, function handle 을 만든다 nlj(p) 형태로 사용할 수 있다.
nlj = @(p) [negLogJoint(r, prc_fun, obs_fun, p(1:n_prcpars), p(n_prcpars+1:n_prcpars+n_obspars))];

% Check whether priors are in a region where the objective function can be evaluated
[dummy1, dummy2, rval, err] = nlj(init);
if rval ~= 0
    rethrow(err);
end

% The objective function is now the negative log joint restricted
% with respect to the parameters that are not optimized
% Obj fx 인 nlj 를 free parameter (오메가, 제타) 를 제외한 나머지 parameter 를
% 고정한 상태로 구하는 function handle 을 만든다
obj_fun = @(p_opt) restrictfun(nlj, init, opt_idx, p_opt); %  restrictfun(nlj, init, opt_idx, init(opt_idx)');

% Optimize 위의 function handle 들을 이용하여, restricted nlj 를  optimize 한다
% 방법은 Quasi_Newton 으로
% 이를 위해서는 restricted nlj 를 구하는 function handle, free parameter,
% optimization 에 대한 option (r.c_opt) 이 필요하다.
disp(' ')
disp('Optimizing...')
r.optim = opt_algo(obj_fun, init(opt_idx)', r.c_opt); % restrictfun 에 의해서 
% 가운데의 init param 중 init(opt_idx)' 부분만 계속 바뀌어서 (tapas_quasi_newton 에서) 
% optimization 되게 된다

% Replace optimized values in init with arg min values
final = init;
final(opt_idx) = r.optim.argMin'; % init 중에서 optimization 된 free parameter 만 replace
r.optim.final = final;

% Get the negative log-joint and negative log-likelihood
[negLj, negLl] = nlj(final);

% Calculate the covariance matrix Sigma and the log-model evidence (as approximated
% by the negative variational free energy under the Laplace assumption).
disp(' ')
disp('Calculating the log-model evidence (LME)...')
d     = length(opt_idx);
Sigma = NaN(d);
Corr  = NaN(d);
LME   = NaN;

options.init_h    = 1;
options.min_steps = 10;

% Numerical computation of the Hessian of the negative log-joint at the MAP estimate
H = tapas_riddershessian(obj_fun, r.optim.argMin, options);

% Use the Hessian from the optimization, if available,
% if the numerical Hessian is not positive definite
if any(isinf(H(:))) || any(isnan(H(:))) || any(eig(H)<=0)
    if isfield(r.optim, 'T')
        % Hessian of the negative log-joint at the MAP estimate
        H     = inv(r.optim.T);
        % Parameter covariance 
        Sigma = r.optim.T;
        % Parameter correlation
        Corr = tapas_Cov2Corr(Sigma);
        % Log-model evidence ~ negative variational free energy
        LME   = -r.optim.valMin + 1/2*log(1/det(H)) + d/2*log(2*pi);
    else
        disp('Warning: Cannot calculate Sigma and LME because the Hessian is not positive definite.')
    end
else
    % Parameter covariance
    Sigma = inv(H);
    % Parameter correlation
    Corr = tapas_Cov2Corr(Sigma);
    % Log-model evidence ~ negative variational free energy
    LME = -r.optim.valMin + 1/2*log(1/det(H)) + d/2*log(2*pi);
end

% Calculate accuracy and complexity (LME = accu - comp)
accu = -negLl;
comp = accu -LME;

% Calculate AIC and BIC
if ~isempty(r.y)
    ndp = sum(~isnan(r.y(:,1)));
else
    ndp = sum(~isnan(r.u(:,1)));
end
AIC  = 2*negLl +2*d;
BIC  = 2*negLl +d*log(ndp);

r.optim.H     = H;
r.optim.Sigma = Sigma;
r.optim.Corr  = Corr;
r.optim.negLl = negLl;
r.optim.negLj = negLj;
r.optim.LME   = LME;
r.optim.accu  = accu;
r.optim.comp  = comp;
r.optim.AIC   = AIC;
r.optim.BIC   = BIC;

return;