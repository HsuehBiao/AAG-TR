
clc; clear;
warning off;
addpath(genpath('./'));

%% dataset
ds={'Caltech256'};
num_runs = 1;

for dsi =1:length(ds)
    dataName = ds{dsi};
    disp(['Processing: ', dataName]);
    % ========== 加载数据集 ==========
    try
        % 假设数据集文件名为 [dataName].mat（如 MFeat_2Views.mat）
        load(fullfile('dataset', [dataName '.mat']));
        % 确保数据变量名正确（这里假设是 X 和 Y）
        % 如果不是，需要调整，例如：X = data.features; Y = data.labels;
    catch
        error('无法加载数据集: %s', dataName);
    end
    % X = X';
    % Y = Y';

    k = length(unique(Y)) ;
    v = length(X);

    iter = 150;

    %% ========== 定义 alpha 和 gamma 的搜索范围 ==========
    % alpha_exponents = -4:1:4;  % 指数范围 [-4, -3, ..., 4]
    % alpha_values = 10.^alpha_exponents;  % [1e-4, 1e-3, ..., 1e4]
    % 
    % gamma_exponents = -4:1:4;  % 同上
    % gamma_values = 10.^gamma_exponents;
    alpha_start = 0.6e-4;
    alpha_end = 1.4e-4;
    alpha_values = linspace(alpha_start, alpha_end, 9);
    gamma_start = 0.6e-3;
    gamma_end = 1.4e-3;
    gamma_values = linspace(gamma_start, gamma_end, 9);

    %% 提前计算 theta 和 target_view
    thetaall = cell(v, 1);
    object_sum = zeros(v, 1);
    
    for iv = 1:v
        [res_neighbor,time_neighbor,label_neighbor,object,theta] = Neighbor(X{iv},Y);
        thetaall{iv,:} = theta;
        object_sum(iv,:) = sum(object);
    end

    [~,target_view] = min(object_sum);

    %% ========== 网格搜索调参 ==========
    best_acc = 0;
    best_alpha = 0;
    best_gamma = 0;
    best_result = zeros(1, 8);  % 存储最佳指标
    alpha_lenghth = length(alpha_values);
    gamma_lenghth = length(gamma_values);
    acc_matrix = zeros(alpha_lenghth, gamma_lenghth);

    %%
    for ia =  1:length(alpha_values)
        for ig = 1:length(gamma_values)
            alpha = alpha_values(ia);
            gamma = gamma_values(ig);

            acc_results = zeros(num_runs, 1);

            U = aas_etlr(X, Y, thetaall, iter, target_view, alpha, gamma);
            [result, ~] = myNMIACCwithmean(U, Y, k);
            acc_matrix(ia,ig) = result(1);
            if result(1) > best_acc
                best_acc = result(1);
                best_alpha = alpha;
                best_gamma = gamma;
                best_result = result;
            end
            fprintf('Alpha: %.2e \t Gamma: %.2e \t ACC: %.4f\n', alpha, gamma, result(1));
        end
    end
    %% ========== 输出最佳结果 ==========
    fprintf('\n===== 最佳参数组合 =====\n');
    fprintf('Alpha: %.3e \t Gamma: %.3e \n', best_alpha, best_gamma);
    fprintf('ACC: %.4f \t NMI: %.4f \t Purity: %.4f \t Fscore: %.4f \n', ...
        best_result(1), best_result(2), best_result(3), best_result(4));
    % 提供的颜色值
    custom_colors = [
        56, 81, 161;
        84, 122, 184;
        115, 172, 208;
        169, 207, 228;
        209, 236, 231;
        239, 245, 199;
        255, 235, 164;
        251, 197, 122;
        245, 149, 88;
        234, 97, 63;
        203, 44, 49;
        170, 2, 35
        ];
    % 归一化颜色值到[0, 1]范围
    custom_colors = custom_colors / 255;
    % 反转颜色值顺序
    custom_colors = flipud(custom_colors);
    % 创建自定义颜色映射
    custom_colormap = flipud(custom_colors);
    % ==== 绘制柱状图 ====
    figure;
    bar3(acc_matrix);
    xlabel('\gamma');
    ylabel('\alpha');
    zlabel('ACC');
    xticklabels(gamma_values);
    yticklabels(alpha_values);
    xticks(1:length(alpha_values));
    yticks(1:length(gamma_values));
    colormap(custom_colormap);

    % ==== 保存图像 ====
    output_folder = 'Parameter_Analysis';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % 构造文件名，使用当前数据集名称 + '_of_parameter'
    filename_base = fullfile(output_folder, [dataName '_of_parameter']);

    % 保存为 PNG 和 EPS 格式
    % 使用 exportgraphics 保存为 PNG（300 DPI）
    exportgraphics(gcf, [filename_base '.png'], 'Resolution', 300);

    % 使用 exportgraphics 保存为 EPS（矢量图，无需指定分辨率）
    exportgraphics(gcf, [filename_base '.pdf']);
    savefig(gcf, [filename_base '.fig']);
end


