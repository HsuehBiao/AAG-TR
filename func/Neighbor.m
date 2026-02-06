function [res_neighbor,time_neighbor,label_neighbor,object,theta,class] = Neighbor(X0,Y)
% Neighbor: 主函数，用于执行层次二分邻居聚类算法并评估聚类结果。
% 输入:
%   X0: 输入数据矩阵，大小为 n x D，其中 n 是样本数量，D 是特征维度。
%   Y: 真实的类别标签，大小为 n x 1，用于评估聚类结果。
% 输出:
%   res_neighbor: 聚类结果的评估指标（如 ACC、NMI 等）。
%   time_neighbor: 聚类算法的运行时间。
%   label_neighbor: 聚类结果的类别标签，大小为 n x 1。
%   object: 每个类别的目标值，大小为 class x 1。
%   theta: 每个类别的中心点，大小为 class x D。
%   class: 该视图的锚点个数。
k = length(unique(Y)) ;
n = size(Y,1);
label = zeros(n,1);
 tic 
 [label_pre,object_pre,theta_pre,num_class_pre,class_pre] = Pre_HBNC(X0);
 [label,object,theta,num_class,class] = Impro_HBNC(X0,label_pre,object_pre);
%%  Organize label results

label_neighbor = label;
res_neighbor = Clustering8Measure(label_neighbor, Y);
time_neighbor = toc;

end

