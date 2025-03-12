function [error_primal, error_dual, gap_function, x_bar, y_bar ] = ALPD_quad_obj_linear_cons_lg(n,m,K,Q,c,A,b,l_yy,l_xy,l_xx,D_x, D_y, l_f, l_g, mu_f, mu_g, x_init,y_init)
x = zeros(n,K+1); 
y = zeros(m,K+1); 
x(:,1) = x_init;    % x_0 as the starting point in primal 
y(:,1) =  y_init;    % y_0 as the starting point in dual 
eta = zeros(1,K+1); % this is 1/eta or 1/tau in chambolle's paper
tau = zeros(1,K+1);  
theta= zeros(1,K+1); 
gamma = zeros(1,K+1); 
gamma(1) = 1; 
v = zeros(n,K+1); 
v(:,1) = A*x(:,1); 
x_bar = zeros(n,K+1); 
x_bar(:,1) = x(:,1); 
y_bar = zeros(m,K+1); 
y_bar(:,1) = y(:,1); 
beta_inv = zeros(1,K+1); 
beta_inv(1) = 1; 
beta = zeros(1,K+1);
error_primal = zeros(1,K);
error_dual = zeros(1,K);
gap_function = zeros(1,K);
gap_arg_prime = zeros(n,K); 
gap_arg_dual = zeros(m,K);
beta(1) = 1;  
x_underbar = zeros(n,K+1); 
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
%% Performiong ALPD algorithm
for t=1:K
    gamma(t+1) = (t+1)/2+(2*sqrt(2)*l_yy+l_g)/mu_g;
    theta(t+1) = gamma(t)/gamma(t+1);
    beta(t+1) = 1+theta(t+1)*beta(t); 
    beta_inv(t+1) = 1/beta(t+1);  
    eta(t) = (l_f+l_xy^2/mu_g)/(t+1)+l_xx; 

    tau(t) = mu_g*t+ 2*sqrt(2)*l_yy+ 2*l_g;
    x_underbar(:,t) = (1-beta_inv(t))* x_bar(:,t)+ beta_inv(t)*x(:,t); 
    cvx_begin quiet  %optiimizaing in dual 
        variable  y_sol(m)
        %minimize (-y_sol'*v(:,t)+ y_sol'*b+ (0.5*mu_g)*sum_square_abs(y_sol) + tau(t)/2*sum_square_abs(y_sol-y(:,t))); %exact dual
         minimize (-y_sol'*v(:,t)+ y_sol'*b+ (mu_g)*y_sol'*y(:,t) + tau(t)/2*sum_square_abs(y_sol-y(:,t)));
        subject to 
           %norm(y_sol)<=D_y;
           norm(y_sol,1)<=D_y;
            %norm(y_sol,Inf)<=D_y;
    cvx_end
        y(:,t+1) = y_sol; %return the optimal solution in dual subproblem.

    cvx_begin quiet %optimizing in primal 
        variable x_sol(n)
        minimize(x_sol'*Q*x_underbar(:,t)+c'*x_sol + x_sol'*A'*y(:,t+1) + eta(t)/2*sum_square_abs(x_sol-x(:,t)));
        subject to 
        norm(x_sol) <= D_x;
        %norm(x_sol,1) <= D_x;
        %norm(x_sol,Inf) <= D_x;
    cvx_end
    x(:,t+1) = x_sol; 
    v(:,t+1) = A*x(:,t+1) + theta(t+1)*(A*x(:,t+1)- A*x(:,t));
    x_bar(:,t+1) = (1-beta_inv(t))* x_bar(:,t)+ beta_inv(t)*x(:,t+1);
    y_bar(:,t+1) = (1-beta_inv(t))* y_bar(:,t)+ beta_inv(t)*y(:,t+1);
     %% Measures 
    error_primal(t) = norm(x_bar(:,t+1)-opt_sol_prim)/norm(opt_sol_prim); 
    error_dual(t) = norm(y_bar(:,t+1)-opt_sol_dual)/norm(opt_sol_dual); 
cvx_begin quiet  % evaluating gap function (dual value)
    variable y_sol(m)
    maximize(0.5*x_bar(:,t+1)'*Q*x_bar(:,t+1)+c'*x_bar(:,t+1)+(A*x_bar(:,t+1)-b)'*y_sol - (0.5*mu_g)*sum_square_abs(y_sol));
    subject to 
        %norm(y_sol)<= D_y ;
        norm(y_sol,1)<= D_y ;
        %norm(y_sol,Inf)<= D_y ;
    cvx_end; 
    opt_value_dual = cvx_optval;
    gap_arg_dual(:,t+1) = y_sol;
    cvx_begin quiet  % evaluating gap function (primal value)
    variable x_sol(n)
    minimize(0.5*x_sol'*Q*x_sol+c'*x_sol+(A*x_sol-b)'*y_bar(:,t+1) - (0.5*mu_g)*sum_square_abs(y_bar(:,t+1)));
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