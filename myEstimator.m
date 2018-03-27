function [x_hat_array, y_hat_array, x_apriori_save] = myEstimator(A,B,H,u,y,x0,Q,R)

x_hat_array = x0;
x_hat = x0;
y_hat_array = [];
P_before = eye(length(A));      % p的初值不能为0

x_hat_priori_array = [];
for i=2:length(y)
    %%%%%%%%%%%%%%% 5 equations %%%%%%%%%%%%%%%
    % predict
    x_hat_priori = A * x_hat + B * u(i-1);
    P_priori =  A* P_before * A' + Q;
    % measurement
    K_k = P_priori * H'/(H * P_priori * H' + R);
    x_hat = x_hat_priori + K_k * (y(:,i) - H * x_hat_priori);
    P_before = (eye(length(A)) - K_k * H) * P_priori;
    %%%%%%%%%%%%%%% 5 equations %%%%%%%%%%%%%%%
    
    % save the data
    x_hat_priori_array = [x_hat_priori_array, x_hat_priori];
    x_hat_array = [x_hat_array x_hat]; %#ok<AGROW>
    y_hat_array = [y_hat_array, (H*x_hat)]; %#ok<AGROW>

end
x_apriori_save = x_hat_priori_array;
end

