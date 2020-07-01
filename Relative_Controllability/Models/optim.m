function r = optim(r, prc_fun, obs_fun, opt_algo)

% Use means of priors as starting values for optimization for optimized parameters (and as values
% for fixed parameters)
init = [r.c_prc.priormus, r.c_obs.priormus]; % ��� prior parameter �� mu(log sa �� ���� sigma>0 �̾�� �ϹǷ�), 
%rho (���� ������ ���� �ʿ� ����), kappa, omega
% level = 3 �� default �̸� omega �� level 1 �� NaN, 3 �� theta �� �ش��Ѵ�, ���� ���
% ��쿡�� level 1�� deterministic �ϹǷ� (level 2�� ���� ����), mean/std �� NaN ���� ����

% Determine indices of parameters to optimize (i.e., those that are not fixed or NaN)
% � parameter �� free �� �� ���ΰ�? �⺻�����δ� omega (2) �� zeta(response model) �� free�� �δµ�
opt_idx = [r.c_prc.priorsas, r.c_obs.priorsas];
opt_idx(isnan(opt_idx)) = 0;
opt_idx = find(opt_idx); % free parameter �� index

% Number of perceptual and observation parameters 
% perceptual, observational model ������ �Ķ���� ����
n_prcpars = length(r.c_prc.priormus);
n_obspars = length(r.c_obs.priormus);

% Construct the objective function to be MINIMIZED:
% The negative log-joint as a function of a single parameter vector
% Objective function �� �����ϰ�, function handle �� ����� nlj(p) ���·� ����� �� �ִ�.
nlj = @(p) [negLogJoint(r, prc_fun, obs_fun, p(1:n_prcpars), p(n_prcpars+1:n_prcpars+n_obspars))];

% Check whether priors are in a region where the objective function can be evaluated
[dummy1, dummy2, rval, err] = nlj(init);
if rval ~= 0
    rethrow(err);
end

% The objective function is now the negative log joint restricted
% with respect to the parameters that are not optimized
% Obj fx �� nlj �� free parameter (���ް�, ��Ÿ) �� ������ ������ parameter ��
% ������ ���·� ���ϴ� function handle �� �����
obj_fun = @(p_opt) restrictfun(nlj, init, opt_idx, p_opt); %  restrictfun(nlj, init, opt_idx, init(opt_idx)');

% Optimize ���� function handle ���� �̿��Ͽ�, restricted nlj ��  optimize �Ѵ�
% ����� Quasi_Newton ����
% �̸� ���ؼ��� restricted nlj �� ���ϴ� function handle, free parameter,
% optimization �� ���� option (r.c_opt) �� �ʿ��ϴ�.
disp(' ')
disp('Optimizing...')
r.optim = opt_algo(obj_fun, init(opt_idx)', r.c_opt); % restrictfun �� ���ؼ� 
% ����� init param �� init(opt_idx)' �κи� ��� �ٲ� (tapas_quasi_newton ����) 
% optimization �ǰ� �ȴ�

% Replace optimized values in init with arg min values
final = init;
final(opt_idx) = r.optim.argMin'; % init �߿��� optimization �� free parameter �� replace
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