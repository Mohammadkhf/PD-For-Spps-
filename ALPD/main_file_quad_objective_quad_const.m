clc;
clear; 
close all;
 %% Setting the parameters
% The problem we defined is based on a high dimensional quadratic objective
% function with a nonlinear constraint 
% The objective function is in the form of f(x) = 0.5*x^T*Q*x + c^Tx where
% Q is a semi definite matrix. c is a n-dimensional normally distributed random vector.
%  The constraint is 0.5*x^T*A_j*x + b_j^Tx+ d where A_j and b_j are produced
%  similar to objective function and d is uniformly distributed between 0
%  and 1 
tStart =tic;
R = 1; % total runs
K = 100; %number of iterations
time_alpd = zeros(1,R);
time_alpd_lg = zeros(1,R);
time_alpd_inexact = zeros(1,R);
time_alpd_inexact_lg = zeros(1,R);
gap_function_ALPD = zeros (R,K);
gap_function_ALPD_lg = zeros (R,K);
gap_function_ALPD_inexact = zeros (R,K/2);
gap_function_ALPD_inexact_lg = zeros (R,K/2);
x_bar_ALPD = cell(1,R);
y_bar_ALPD = cell(1,R);
x_bar_ALPD_lg = cell(1,R);
y_bar_ALPD_lg = cell(1,R);
x_bar_ALPD_inexact= cell(1,R);
y_bar_ALPD_inexact = cell(1,R);
x_bar_ALPD_inexact_lg = cell(1,R);
y_bar_ALPD_inexact_lg = cell(1,R);
iteration_time_ALPD = zeros (R,K);
iteration_time_ALPD_lg = zeros (R,K);
iteration_time_ALPD_inexact = zeros (R,K/2);
iteration_time_ALPD_inexact_lg = zeros (R,K/2);
for r=1:R
n = 100; % set the dimension of the primal  
m = 10;  % set the dimension of the dual 
diagonal_Q = diag(rand(n,1)*100); % constructing diagonal matrix for the objective function
orthonormal_Q = orth(randn(n)); % constructing orthonormal matrix for the objective function
Q = orthonormal_Q'*diagonal_Q*orthonormal_Q; 
c = randn(n,1); 
A = zeros(n,n,m);
B = zeros(n,m);

for j=1:m
  diagonal_A = diag(rand(n,1)*100); % constructing diagonal matrix for the constraint
orthonormal_A = orth(randn(n)); % constructing orthonormal matrix for the constraint
A(:,:,j) = orthonormal_A'*diagonal_A*orthonormal_A; %constructing the tensor containing m of n*n matrices A_j. 
B(:,j) = randn(n,1); %contructng the matrix containing m of n dimensional vectors.  
end
D_x = 1 ; 
D_y =1 ;
norms = zeros(m,1);
norm_A = zeros(m,1); 
for j=1:m
    norms(j) = D_x*norm(A(:,:,j))+ norm(B(:,j));
    norm_A(j) = norm(A(:,:,j)); 
end
d  = rand(m,1); % vector of constants  
l_yy = 0 ; 
l_xy = norm(norms);
l_xx = 20*D_y * norm(norm_A);
l_f = norm(Q); %set the lipschitz smooth constant
l_g = 0 ; %set the lipschitz constant of dual. lg_= 0 if we solve g exactly. 
mu_f = 0 ;         % set the strong convexity modulous for f 
mu_g = 1 ;         % set the strong convexity modulous for g 
z_1 = randn(n,1);
%initialization
cvx_begin quiet %change the constraints based on different norms
    variable x_sol(n)
    minimize(0.5*sum_square_abs(x_sol-z_1));
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
    subject to %change the constraints based on different norms
         0 <= y_sol;
        norm(y_sol)<= D_y;
        %norm(y_sol,1)<= D_y;
        %norm(y_sol,Inf)<= D_y ;
cvx_end; 
y_init = y_sol;
%% Run the algorithms
tstart_ALPD_inexact=tic;
[gap_function_ALPD_inexact(r,:), x_bar_ALPD_inexact{1,r}, y_bar_ALPD_inexact{1,r},iteration_time_ALPD_inexact(r,:)]=ALPD_inexact_quad_obj_quad_cons(n,m,K/2,Q,c,A,B,d,l_yy,l_xy,l_xx, l_f, l_g,mu_f,mu_g,D_x, D_y,x_init,y_init);
time_alpd_inexact (r) = toc(tstart_ALPD_inexact);
tstart_ALPD=tic;
[gap_function_ALPD(r,:), x_bar_ALPD{1,r}, y_bar_ALPD{1,r},iteration_time_ALPD(r,:)]=ALPD_quad_obj_quad_cons(n,m,K,Q,c,A,B,d,l_yy,l_xy,l_xx, l_f, l_g,mu_f,mu_g, D_x, D_y, x_init,y_init);
time_alpd(r) = toc(tstart_ALPD);
% 
l_g = 2000*mu_g ; 
tstart_ALPD_Lg=tic;
[gap_function_ALPD_lg(r,:), x_bar_ALPD_lg{1,r}, y_bar_ALPD_lg{1,r},iteration_time_ALPD_lg(r,:)]=ALPD_quad_obj_quad_cons_lg(n,m,K,Q,c,A,B,d,l_yy,l_xy,l_xx, l_f, l_g, mu_f,mu_g,D_x, D_y, x_init,y_init);
time_alpd_lg(r) = toc(tstart_ALPD_Lg);
tstart_ALPD_inexact_lg=tic;
[gap_function_ALPD_inexact_lg(r,:), x_bar_ALPD_inexact_lg{1,r}, y_bar_ALPD_inexact_lg{1,r},iteration_time_ALPD_inexact_lg(r,:)]=ALPD_inexact_lg_quad_obj_quad_cons(n,m,K/2,Q,c,A,B,d,l_yy,l_xy,l_xx, l_f, l_g,mu_f,mu_g,D_x, D_y,x_init,y_init);
time_alpd_inexact_lg(r) = toc(tstart_ALPD_inexact_lg);
end
%% Average measures
avg_gap_function_ALPD = mean(gap_function_ALPD,1);
avg_gap_function_ALPD_lg = mean(gap_function_ALPD_lg,1);
avg_gap_function_ALPD_inexact = mean(gap_function_ALPD_inexact,1);
avg_gap_function_ALPD_inexact_lg = mean(gap_function_ALPD_inexact_lg,1);
avg_iteration_time_ALPD_inexact = mean(cumsum(iteration_time_ALPD_inexact,2),1);
avg_iteration_time_ALPD = mean(cumsum(iteration_time_ALPD,2),1);
avg_iteration_time_ALPD_lg = mean(cumsum(iteration_time_ALPD_lg,2),1);
avg_iteration_time_ALPD_inexact_lg = mean(cumsum(iteration_time_ALPD_inexact_lg,2),1);
%% pLot
figure('Name','Measures primal');
plot(avg_iteration_time_ALPD,avg_gap_function_ALPD,'color','#f80','LineWidth',2); 
hold on 
plot(avg_iteration_time_ALPD_lg,avg_gap_function_ALPD_lg,'LineWidth',2); 
hold on 
plot(avg_iteration_time_ALPD_inexact,avg_gap_function_ALPD_inexact,'LineWidth',2);
hold on 
plot(avg_iteration_time_ALPD_inexact_lg,avg_gap_function_ALPD_inexact_lg,'--','LineWidth',2);
legend("ALPD-prox-g","ALPD","Inexact-ALPD-prox-g","Inexact ALPD-")
hold off
%title("Gap function convergence")
xlabel('Run time (seconds)','FontWeight','bold')
ylabel('$Gap(\bar{z})$','Interpreter','latex','fontsize',18,'FontWeight','bold')
set(gca,'FontWeight','bold') 
%%
tEnd = toc(tStart);