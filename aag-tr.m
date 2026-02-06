function [UU,err_history] = aas_etlr(X, Y, theta, max_iter, target_view,alpha,gamma)
    %% 初始化
    numview = length(X);
    N = size(Y, 1);
    c = size(theta{target_view}, 1);%最优视图的锚点数量
    % c=5;
    J_tensor = zeros(N, c, numview);
    J_prev = cell(1, numview);
    % alpha = 1e1; gamma = 1e-1; 
    mu = 1e-5; max_mu = 1e6; pho_mu = 1.2; tol = 1e-8;
    
    % 预分配变量
    for k = 1:numview
        d = size(theta{k}, 1);
        Z{k} = orth(randn(N, d));      % N×d，正交约束
        G{k} = orth(randn(N, c));      % N×c，正交约束
        J_tensor(:, :, k) = G{k};
        J_prev{k} = J_tensor(:, :, k);
        P{k} = randn(c, d);            % c×d，
        H{k} = randn(d, size(X{k}, 2)); % d×D_k
        y{k} = zeros(N, c);            % 拉格朗日乘子
    end
    err_history = [];
    %% ADMM主循环
    for iter = 1:max_iter
        fprintf('----Iteration %d----\n', iter);
        J_cell = cell(1, numview);
        for k = 1:numview
            J_cell{k} = J_tensor(:, :, k);  % 使用当前 J_tensor 的切片
        end
        %% 并行更新局部变量 Z^v, P^v, H^v
        for k = 1:numview  
            % 更新 Z^v 
            Mz = alpha * G{k} * P{k} + 2 * gamma * X{k} * H{k}';
            [Uz, ~, Vz] = svd(Mz, 'econ');
            Z{k} = Uz * Vz';
            
            % 更新 P^v 
            Mp = alpha * G{k}' * Z{k};
            [Up, ~, Vp] = svd(Mp, 'econ');
            P{k} = Up * Vp';
            
            % 更新 H^v 
            H{k} = -2 * gamma * Z{k}' * X{k};

            % (d) 更新G^v
            Mg = alpha * Z{k} * P{k}' + mu * J_cell{k} - y{k};
            [Ug, ~, Vg] = svd(Mg, 'econ');
            G{k} = Ug * Vg'; 
        end
        
        %% 全局更新 G^v (ETR约束)
        G_tensor = cat(3, G{:});  % 组合为N×c×V张量
        J_tensor = solve_G(G_tensor + (1/mu) * cat(3, y{:}), mu, [N, c, numview], 1e-2);
        % [out_put,~] = wshrinkObj(G_tensor + (1/mu) * cat(3, y{:}),1/mu,[N,c,numview],0,2);
        % J_tensor = reshape(out_put,[N,c,numview]);

        %% 3. 更新拉格朗日乘子y^v
        for k = 1:numview
            y{k} = y{k} + mu * (G{k} - J_tensor(:, :, k));
        end
        
        %% 收敛检查
        primal_err = 0;
        dual_err = 0;
        for k = 1:numview
            primal_err = max(primal_err, norm(G{k} - J_tensor(:, :, k), 'fro'));
            dual_err = max(dual_err, norm(J_tensor(:, :, k) - J_prev{k}, 'fro') );
        end
        err = max([primal_err,dual_err]);
        fprintf('Iter=%d, pri_err=%.8f, dual_err=%.8f, Error=%.8f\n', iter, primal_err, dual_err, err);

        err_history = [err_history; err];

        if err < tol
            break;
        end
        for k = 1:numview
            J_prev{k} = J_tensor(:, :, k);
        end
        %% 更新惩罚参数
        mu = min(mu * pho_mu, max_mu);
    end
    
    %% 构建相似度矩阵并谱聚类
    S = zeros(N);
    for k = 1:numview
        S = S + G{k} * G{k}';
    end
    S = 0.5 * (S + S');
    [UU, ~] = eigs(S, c, 'la');

    %     %% 绘制 err 曲线图
    % figure;
    % plot(1:length(err_history), err_history, '-o', 'LineWidth', 1.5);
    % xlabel('Iteration');
    % ylabel('Error');
    % title('Convergence of Error');
    % grid on;
end