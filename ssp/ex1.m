#!/usr/bin/octave -q

# Two ways to run the program:
# 1. chmod +x ex1.m && ./ex1.m
# 2. octave -ef ex1.m

# Statistical Signal Processing
# Exercise 1 
# Author: Sagi Kedmi
# Date:   14/11/2014

function [A_MMSE_LS A_MMSE_WLS E_MMSE_LS E_MMSE_WLS] = fn_mmse(n)
    theta=1;
    H=(1:n)'; % H=[1,2,...,n]'
    N = 555; % Number of trials

    % Covariance matrix of V,Foreach i: E[Vi]=0, E[ViVi]=1, i!=j E[ViVj] = 0.5 
    covV = 0.5*(eye(n)+ones(n,n));
    
    % Analytical MMSE LS/WLS
    % By the way: inv(H'*H)*H' == pinv(H)
    A_MMSE_LS = pinv(H)*covV*(pinv(H)');   
    
    % Gauss Markov Theorem - BLUE - Best Linear Unbiased Estimator - 
    % States that the WLS estimator with W=inv(covV) is the Minimum Squared
    % Error (MMSE) estimator among all affine unbiased estimators of Q(Theta) 
    % given y (y ~HQ). [Conditions: H - full rank, E[v]=0, covV > 0 (PSD)]
    W = inv(covV);
    G = inv(H'*W*H)*(H')*W;
    A_MMSE_WLS = G*covV*(G');

    mse_ls_sum=0; 
    mse_wls_sum=0;
    for i=1:N

        % Draw n random d-dimensional vectors from a multivariate 
        % Gaussian distribution with mean mu(nxd) and covariance matrix Sigma(dxd).
    
        % Return a matrix with normally distributed random elements having zero mean and variance one.
        X = randn(1,n);
        % http://math.stackexchange.com/questions/163470/generating-correlated-random-numbers-why-does-cholesky-decomposition-work
        % The idea is: E[(LX)(LX)']=L*E[XX']*L'=L*I*L=covV
        L = chol(covV,"lower");
        R = L*(X');
        y = H + R;

        est_theta_ls = pinv(H)*y;
        est_theta_wls =G*y;

        err_ls = est_theta_ls - theta;        
        err_wls = est_theta_wls - theta;        
        
        mse_ls_sum+= err_ls*err_ls';
        mse_wls_sum+= err_wls*err_wls';
    end    
 
    E_MMSE_LS = mse_ls_sum/N;
    E_MMSE_WLS = mse_wls_sum/N;
endfunction

for i=2:2:20
    [A_MMSE_LS(i/2) A_MMSE_WLS(i/2) E_MMSE_LS(i/2) E_MMSE_WLS(i/2)] = fn_mmse(i);
    T(i/2)=i;
end

subplot(2,1,1);
plot(T,A_MMSE_LS,'r','LineWidth',2,T,A_MMSE_WLS,'b','LineWidth',2);
set(gca,'xtick',2:2:20);
xlabel("N");
ylabel("MMSE")
grid on;
legend("LS-Analytical", "WLS-Analytical");

subplot(2,1,2);
plot(T,E_MMSE_LS,'r-','LineWidth',2,T,E_MMSE_WLS,'b-','LineWidth',2);
set(gca,'xtick',2:2:20);
xlabel("N");
ylabel("MMSE")
grid on;
legend("LS-Empirical", "WLS-Empirical");

print -dpng "ex1.png"
