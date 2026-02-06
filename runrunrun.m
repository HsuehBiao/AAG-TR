
clc; clear;
%warning off;
addpath(genpath('./'));

%% dataset
ds={'BDGP_fea'};

iter = 120;
alpha=1e-4;
gamma=7e-5;
num_runs = 1; % 运行次数

for dsi =1:length(ds)
    dataName = ds{dsi};
    disp(['Processing: ', dataName]);
    all_results = zeros(num_runs, 8); % 存储每次运行的8个指标
    all_times = zeros(num_runs, 1);   % 存储每次运行的时间
    tsne_results = cell(num_runs, 1); % 存储每次运行的 t-SNE 结果

    for run = 1:num_runs
        disp(['Run ', num2str(run), '/', num2str(num_runs)]);
        % ========== 加载数据集 ==========
        try
            % 假设数据集文件名为 [dataName].mat（如 MFeat_2Views.mat）
            load(fullfile('dataset', [dataName '.mat']));
            % 确保数据变量名正确（这里假设是 X 和 Y）
            % 如果不是，需要调整，例如：X = data.features; Y = data.labels;
        catch
            error('无法加载数据集: %s', dataName);
        end
        X = X';
        % Y = Y';


        k = length(unique(Y));
        v = length(X);
        %% 初始化变量
        thetaall = cell(v, 1);       % 显式初始化
        object_sum = zeros(v, 1);    % 预分配内存
        %%
        t_start = tic;
        for iv = 1:v
            [res_neighbor,time_neighbor,label_neighbor,object,theta] = Neighbor(X{iv},Y);
            thetaall{iv,:} = theta;
            object_sum(iv,:) = sum(object);
        end

        [~,target_view] = min(object_sum);

        [U,err_history] = aas_etlr(X,Y,thetaall,iter,target_view,alpha,gamma);
        
        % em = tsne(U,"Perplexity",30,'Standardize',true);
    

        [result,~] = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
        times  = toc(t_start);
        % 存储结果
        all_results(run, :) = result;
        all_times(run) = times;

        fprintf('=== Results (α=%.0e, γ=%.0e) ===\n', alpha, gamma);
        fprintf('ACC: %.4f \t NMI: %.4f \t Purity: %.4f \t Fscore: %.4f \t Time: %.2fs\n', ...
            result(1), result(2), result(3), result(4),times);
    end
    avg_result = mean(all_results, 1);
    avg_time = mean(all_times);
    fprintf('=== Average Results (after %d runs, α=%.0e, γ=%.0e) ===\n', num_runs, alpha, gamma);
    fprintf('ACC: %.4f \t NMI: %.4f \t Purity: %.4f \t Fscore: %.4f \t Time: %.2fs\n', ...
        avg_result(1), avg_result(2), avg_result(3), avg_result(4), avg_time);
    
    %tsne
    % figure;
    % gscatter(em(:,1),em(:,2),Y,[],[],12);
    % legend('Location', 'southeast'); 
    % grid on;
    % imgFolder = 'TSNE_Plots';  % 指定保存文件夹
    % if ~exist(imgFolder, 'dir')
    %     mkdir(imgFolder);      % 自动创建文件夹（如果不存在）
    % end
    % filename = fullfile(imgFolder, 'BDGP tsne.pdf');
    % exportgraphics(gcf, filename, 'Resolution', 300);

    %Convergence
    % resultFolder = 'Convergence';
    % if ~exist(resultFolder, 'dir')
    %     mkdir(resultFolder);
    % end
    % figure;
    % plot(1:length(err_history), err_history, '-o', 'LineWidth', 1.5);
    % xlabel('Iteration Number');
    % ylabel("Error");
    % % title(['Convergence of ', dataName]);
    % grid on;
    % fullname = fullfile(resultFolder,['convergence of ',dataName,'.pdf']);
    % exportgraphics(gcf,fullname);
end


