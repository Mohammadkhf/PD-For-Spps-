clc;
clear; 
close all;
 %% Setting the parameters
% The problem we defined is based on a high dimensional quadratic objective
% function with linear constraints
% The objective function is in the form of f(x) = 0.5*x^T*Q*x + c^Tx where
% Q is a semi definite matrix. c is a n-dimensional normally distributed random vector.
%  The constraints have the form of Ax-b<=0 where A and b are drawn from 
% uniform(0,1). 
tStart =tic;
R = 10; % total runs
K = 10; %number of iterations
time_alpd = zeros(1,R);
time_lpd = zeros(1,R);
time_alpd_lg = zeros(1,R);
error_primal_ALPD  = zeros(R,K);
error_dual_ALPD = zeros (R,K);
gap_function_ALPD = zeros (R,K);
x_bar_ALPD = cell(1,R);
y_bar_ALPD = cell(1,R);
error_prime_LPD = zeros(R,K);
error_dual_LPD = zeros(R,K);
gap_function_LPD = zeros (R,K);
x_bar_PD = cell(1,R);
y_bar_LPD = cell(1,R);
error_primal_ALPD_lg = zeros(R,K);
error_dual_ALPD_lg = zeros(R,K);
gap_function_ALPD_lg = zeros (R,K);
x_bar_ALPD_lg = cell(1,R);
y_bar_ALPD_lg = cell(1,R);
for r=1:R
n = 100; % set the dimension of the primal  
m = 100;  % set the dimension of the dual 
diagonal = diag(rand(n,1)*100); % constructing diagonal matrix
orthonormal = orth(randn(n)); % constructing orthonormal matrix 
Q = orthonormal'*diagonal*orthonormal; 
c = randn(n,1); 
A = rand(m,n); % Ax<=b
b  = rand(m,1);
l_yy = 0 ; 
l_xy = norm(A);
l_xx = 0;
l_f =2*norm(Q); %set the lipschitz smooth constant
l_g = 0 ; %set the lipschitz constant of dual. lg_= 0 if we solve g exactly. 
mu_f = 0 ;         % set the strong convexity modulous for f 
mu_g = 1 ;         % set the strong convexity modulous for g 
D_x = 1 ; 
D_y =1 ; 
z_1 = randn(n,1);
cvx_begin quiet
    variable x_sol(n)
    minimize(0.5*sum_square_abs(x_sol-z_1));
    %change the constraints w.r.t different norms
    subject to 
        norm(x_sol)<= D_x ;
        %norm(x_sol,1)<= D_x ;
        %norm(x_sol,Inf)<= D_x ;
cvx_end; 
x_init = x_sol; 
z_2 = randn(m,1);
cvx_begin quiet
    variable y_sol(m)
    minimize(0.5*sum_square_abs(y_sol-z_2));
    %change the constraints w.r.t different norms
    subject to 
        norm(y_sol)<= D_y;
        %norm(y_sol,1)<= D_y;
        %norm(y_sol,Inf)<= D_y ;
cvx_end; 
y_init = y_sol;
tstart_ALPD=tic;
[error_primal_ALPD(r,:), error_dual_ALPD(r,:), gap_function_ALPD(r,:), x_bar_ALPD{1,r}, y_bar_ALPD{1,r}]=ALPD_quad_obj_linear_cons(n,m,K,Q,c,A,b,l_yy,l_xy,l_xx,D_x,D_y, l_f, l_g, mu_f,mu_g,x_init,y_init);
time_alpd(r) = toc(tstart_ALPD);
tstart_LPD=tic; 
[error_prime_LPD(r,:), error_dual_LPD(r,:), gap_function_LPD(r,:), x_bar_PD{1,r}, y_bar_LPD{1,r}] = LPD_quad_obj_linear_cons(n,m,K,Q,c,A,b,D_x,D_y, l_f, mu_f,mu_g,x_init,y_init);
time_lpd(r) = toc(tstart_LPD);
l_g = mu_g ; 
tstart_ALPD_Lg=tic;
[error_primal_ALPD_lg(r,:), error_dual_ALPD_lg(r,:), gap_function_ALPD_lg(r,:), x_bar_ALPD_lg{1,r}, y_bar_ALPD_lg{1,r}]=ALPD_quad_obj_linear_cons_lg(n,m,K,Q,c,A,b,l_yy,l_xy,l_xx,D_x,D_y, l_f, l_g, mu_f,mu_g,x_init,y_init);
time_alpd_lg(r) = toc(tstart_ALPD_Lg);
end
%% Average measures
avg_error_primal_ALPD = mean(error_primal_ALPD);
avg_error_prime_LPD = mean(error_prime_LPD);
avg_error_primal_ALPD_lg = mean(error_primal_ALPD_lg);
avg_error_dual_ALPD = mean(error_dual_ALPD);
avg_error_dual_LPD = mean(error_dual_LPD);
avg_error_dual_ALPD_lg = mean(error_dual_ALPD_lg);
avg_gap_function_ALPD = mean(gap_function_ALPD);
avg_gap_function_LPD = mean(gap_function_LPD);
avg_gap_function_ALPD_lg = mean(gap_function_ALPD_lg);
%% pLot
figure('Name','Measures primal');
subplot(2,2,1); 
plot((K/2:K),avg_error_primal_ALPD(K/2:K),'LineWidth',2); 
hold on 
plot((K/2:K),avg_error_prime_LPD(K/2:K),'LineWidth',2);
hold on 
plot((K/2:K),avg_error_primal_ALPD_lg(K/2:K),'LineWidth',2);
legend("ALPD-prox-g","LPD","ALPD")
hold off
%title("Primal convergence");
xlabel('# of iterations','FontWeight','bold')
ylabel('$\frac{\|\bar{x}_t-x^*\|}{\|x^*\|}$','Interpreter','latex','fontsize',20,'FontWeight','bold')
set(gca,'FontWeight','bold') 
subplot(2,2,2);
%figure('Name','Measures Dual')
plot((K/2:K),avg_error_dual_ALPD(K/2:K),'LineWidth',2);
hold on 
plot((K/2:K),avg_error_dual_LPD(K/2:K),'LineWidth',2);
hold on 
plot((K/2:K),avg_error_dual_ALPD_lg(K/2:K),'LineWidth',2);
legend("ALPD-prox-g","LPD","ALPD")
hold off
%title("Dual convergence");
xlabel('# of iterations','FontWeight','bold')
ylabel('$\frac{\|\bar{y}_t-y^*\|}{\|y^*\|}$','Interpreter','latex','fontsize',20,'FontWeight','bold')
set(gca,'FontWeight','bold') 
%figure('Name','Gap function')
 subplot(2,2,[3,4]);
plot((K/2:K),avg_gap_function_ALPD(K/2:K),'LineWidth',2); 
hold on 
plot((K/2:K),avg_gap_function_LPD(K/2:K),'LineWidth',2); 
hold on 
plot((K/2:K),avg_gap_function_ALPD_lg(K/2:K),'LineWidth',2); 
legend("ALPD-prox-g","LPD","ALPD")
hold off
%title("Gap function convergence")
xlabel('# of iterations','FontWeight','bold')
ylabel('$Gap(\bar{z})$','Interpreter','latex','fontsize',18,'FontWeight','bold')
set(gca,'FontWeight','bold') 
%%
tEnd = toc(tStart);