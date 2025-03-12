function [ gap_function, x_bar, y_bar,iteration_time ] = ALPD_inexact_quad_obj_quad_cons(n,m,K,Q,c,A,B,d,l_yy,l_xy,l_xx, l_f, l_g,mu_f,mu_g,D_x, D_y, x_init,y_init);
%% Setting the parameters
x = zeros(n,K+1); 
y = zeros(m,K+1); 
x(:,1) = x_init;    % x_0 as the starting point in primal 
y(:,1) =  y_init;    % y_0 as the starting point in dual 
eta = zeros(1,K+1); % this is 1/eta or 1/tau in chambolle's paper
tau = zeros(1,K+1);  
theta= zeros(1,K+1); 
gamma = zeros(1,K+1); 
gamma(1) = 1; 
v = zeros(m,K+1); 
l_xx = 0; 
for j=1:m
v(j,1) = 0.5*x(:,1)'*A(:,:,j)*x(:,1)+B(:,j)'*x(:,1); 
end
x_bar = zeros(n,K+1); 
x_bar(:,1) = x(:,1); 
y_bar = zeros(m,K+1); 
y_bar(:,1) = y(:,1); 
beta_inv = zeros(1,K+1); 
beta_inv(1) = 1; 
beta = zeros(1,K+1);
gap_function = zeros(1,K);
iteration_time = zeros(1,K);
gap_arg_prime = zeros(n,K); 
gap_arg_dual = zeros(m,K);
beta(1) = 1; 
x_underbar = zeros(n,K+1); 
%% Performiong Inexact ALPD algorithm
for t=1:K
    tStart_iteration_ALPD = tic; 
    gamma(t+1) = (t+1)/2+(2*sqrt(2)*l_yy+2*l_g)/mu_g;
    theta(t+1) = gamma(t)/gamma(t+1);
    beta(t+1) = 1+theta(t+1)*beta(t); 
    beta_inv(t+1) = 1/beta(t+1);  
    eta(t) = (l_f+l_xy^2/mu_g)/(t+1)+l_xx; 
    %eta(t) = 1/eta(t); 
    tau(t) = mu_g*t/2+ 2*sqrt(2)*l_yy+ 2*l_g;
    x_underbar(:,t) = (1-beta_inv(t))* x_bar(:,t)+ beta_inv(t)*x(:,t); 
    cvx_begin quiet  %optiimizaing in dual 
        variable  y_sol(m)
        %exact dual (prox-g)
        minimize (-y_sol'*v(:,t)+ y_sol'*d+ (0.5*mu_g)*sum_square_abs(y_sol) + tau(t)/2*sum_square_abs(y_sol-y(:,t)));
        %approx dual (linearize g)
        %minimize (-y_sol'*v(:,t)+ y_sol'*d+ (mu_g)*y_sol'*y(:,t) +
        %tau(t)/2*sum_square_abs(y_sol-y(:,t))); 
        % change the constrants based on the norms
        subject to 
            0 <= y_sol;
            norm(y_sol)<=D_y;
            %norm(y_sol,1)<=D_y;
            %norm(y_sol,inf)<=D_y;
    cvx_end
        y(:,t+1) = y_sol; %return the optimal solution in dual subproblem.
    
    tensor = zeros(n);
        for j=1: m
            tensor = tensor + y(j,t+1).* A(:,:,j);
        end    
    B_y = B *y(:, t+1); 
    cvx_begin quiet %optimizing in primal 
    cvx_precision medium
    cvx_solver sedumi
        variable x_sol(n)
        objective  = 0.5*x_sol'*tensor*x_sol + B_y'*x_sol; 
        minimize(x_sol'*Q*x_underbar(:,t)+c'*x_sol + objective + eta(t)/2*sum_square_abs(x_sol-x(:,t)));
        % change the norms based on the problem
        subject to 
        norm(x_sol) <= D_x;
        %norm(x_sol,1) <= D_x;
        %norm(x_sol,inf) <= D_x;
    cvx_end
    x(:,t+1) = x_sol; 
       for j=1:m
    v(j,t+1) = 0.5*x(:,t+1)'*A(:,:,j)*x(:,t+1)+B(:,j)'*x(:,t+1) + theta(t+1)*((0.5*x(:,t+1)'*A(:,:,j)*x(:,t+1)+B(:,j)'*x(:,t+1))- (0.5*x(:,t)'*A(:,:,j)*x(:,t)+B(:,j)'*x(:,t)));
       end
    x_bar(:,t+1) = (1-beta_inv(t))* x_bar(:,t)+ beta_inv(t)*x(:,t+1);
    y_bar(:,t+1) = (1-beta_inv(t))* y_bar(:,t)+ beta_inv(t)*y(:,t+1);
    tEnd_iteration_ALPD = toc(tStart_iteration_ALPD);
    iteration_time(t) = tEnd_iteration_ALPD;

end
for t=1:K 
cvx_begin quiet  % evaluating gap function (dual value)
    variable y_sol(m)
    objective = 0 ;
    for j=1:m 
        objective = objective + (0.5*x_bar(:,t+1)'*A(:,:,j)*x_bar(:,t+1)+B(:,j)'*x_bar(:,t+1)-d(j))*y_sol(j);
   end
    maximize(0.5*x_bar(:,t+1)'*Q*x_bar(:,t+1)+c'*x_bar(:,t+1)+objective- (0.5*mu_g)*sum_square_abs(y_sol));
    subject to 
        0 <= y_sol;
        norm(y_sol)<= D_y; 
    cvx_end; 
    opt_value_dual = cvx_optval;
    gap_arg_dual(:,t) = y_sol;
    cvx_begin quiet % evaluating gap function (primal value)
    variable x_sol(n)
    objective = 0;
     for j=1:m
         objective = objective + (0.5*x_sol'*A(:,:,j)*x_sol+B(:,j)'*x_sol-d(j))*y_bar(j,t+1);
    end
    minimize(0.5*x_sol'*Q*x_sol+c'*x_sol+objective- (0.5*mu_g)*sum_square_abs(y_bar(:,t+1)));
    subject to 
        norm(x_sol)<=D_x; 
    cvx_end; 
    opt_value_primal = cvx_optval;
    gap_arg_prime(:,t) = x_sol; 
    gap_function(t) = opt_value_dual - opt_value_primal; 
end
end