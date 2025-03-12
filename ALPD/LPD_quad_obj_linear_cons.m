function [error_prime, error_dual, gap_function, x_bar, y_bar] = LPD_quad_obj_linear_cons(n,m,K,Q,c,A,b,D_x,D_y, l_f, mu_f,mu_g,x_init,y_init)
%% Setting the parameters
%% Setting the parameters
norm_A_sqr = norm(A)^2; % set the norm of A
x = zeros(n,K+1); 
y = zeros(m,K+1);
x(:,1) = x_init;    % x_0 as the starting point in primal 
y(:,1) =  y_init;    % y_0 as the starting point in dual 
eta = zeros(1,K+1); % this is 1/eta
tau = zeros(1,K+1); %initialization of tau this is 1/tau
theta= zeros(1,K+1); 
gamma = zeros(1,K+1);  
x_tild = zeros(n,K+1);
x_tild(:,1) = x(:,1);
gap_function = zeros(1,K); %gap function avaluating at y_bar and x_bar
gap_arg_prime = zeros(n,K); 
gap_arg_dual = zeros(m,K); 
x_bar = zeros(n,K);
y_bar = zeros(m,K);
error_prime = zeros(1,K);
error_dual = zeros (1,K) ;
 %% Geting the optimal solution in primal 
 cvx_begin quiet
    variable x_sol(n)
    minimize(0.5*x_sol'*Q*x_sol+c'*x_sol+ (0.5/mu_g)*sum_square_abs(A*x_sol-b))
    subject to 
    norm(x_sol)<=D_x;
    %norm(x_sol,1)<=D_x;
    %norm(x_sol,Inf)<=D_x;
 cvx_end;
opt_sol_prim = x_sol; 
opt_value = cvx_optval; 
%% Calculating the saddle point 
cvx_begin quiet
    variable y_sol(m)
    maximize((A*opt_sol_prim-b)'*y_sol - (0.5*mu_g)*sum_square_abs(y_sol));
    subject to 
       %norm(y_sol)<= D_y ;
        norm(y_sol,1)<= D_y ;
        %norm(y_sol,Inf)<= D_y ;
cvx_end; 
opt_sol_dual = y_sol; 
test = A*opt_sol_prim-b; % test our solution with analytical solution.
%% LPD algorithm for the problem and measures 
for t=1:K 
    gamma(t+1) = t+1; 
    gamma(t+2) = t+2;
    theta(t) = gamma(t+1)/gamma(t+2);
    eta(t) = 2*norm_A_sqr/(mu_g*(t+1)) + l_f;
    tau(t) = mu_g*t/2;
    cvx_begin quiet  %optiimizaing in dual 
        variable  y_sol(m)
        minimize (-(A*x_tild(:,t)-b)'*y_sol + (0.5*mu_g)*sum_square_abs(y_sol) + tau(t)/2*sum_square_abs(y_sol-y(:,t)));
        subject to 
           % norm(y_sol)<= D_y ; 
            norm(y_sol,1)<= D_y ; 
            %norm(y_sol,Inf)<= D_y ; 
    cvx_end;
        y(:,t+1) = y_sol; %return the optimal solution in dual subproblem. 
    
    cvx_begin quiet %optimizing in primal 
        variable x_sol(n)
        minimize(x_sol'*Q*x(:,t)+c'*x_sol + (A'*y(:,t+1))'*x_sol + eta(t)/2*sum_square_abs(x_sol-x(:,t)));
        subject to 
        norm(x_sol)<=D_x;
        %norm(x_sol,1)<=D_x;
        %norm(x_sol,Inf)<=D_x;
    cvx_end;
    x(:,t+1) = x_sol; 
    x_tild(:,t+1) = x(:,t+1) + theta(t)*(x(:,t+1)- x(:,t));
    %% Measures and convergence 
    x_bar(:,t) = sum(gamma(2:t+1).*x(:,2:t+1),2)/sum(gamma(2:t+1));
    error_prime(t) = norm(x_bar(:,t)-opt_sol_prim)/norm(opt_sol_prim); 
    y_bar(:,t) = sum(gamma(2:t+1).*y(:,2:t+1),2)/sum(gamma(2:t+1));
    error_dual(t) =  norm(y_bar(:,t)-opt_sol_dual)/norm(opt_sol_dual);
    cvx_begin quiet  % evaluating gap function (dual value)
    variable y_sol(m)
    maximize(x_bar(:,t)'*Q*x_bar(:,t)+c'*x_bar(:,t)+(A*x_bar(:,t)-b)'*y_sol - (0.5*mu_g)*sum_square_abs(y_sol));
    subject to 
        %norm(y_sol)<= D_y ; 
        norm(y_sol,1)<= D_y ; 
        %norm(y_sol,Inf)<= D_y ; 
    cvx_end; 
    opt_value_dual = cvx_optval;
    gap_arg_dual(:,t) = y_sol;
    cvx_begin quiet  % evaluating gap function (primal value)
    variable x_sol(n)
    minimize(x_sol'*Q*x_sol+c'*x_sol+(A*x_sol-b)'*y_bar(:,t) - (0.5*mu_g)*sum_square_abs(y_bar(:,t)));
    subject to 
        norm(x_sol)<=D_x; 
        %norm(x_sol,1)<=D_x; 
        %norm(x_sol,Inf)<=D_x; 
    cvx_end; 
    opt_value_primal = cvx_optval;
    gap_arg_prime(:,t) = x_sol; 
    gap_function(t) = opt_value_dual - opt_value_primal; 
end
end