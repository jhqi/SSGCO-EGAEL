close all;
clear all;
clc;

%% read gray img
dataset = 'BO'; % IP, PU, TR, BO
img_file_path=['pca_img/', dataset, '_pca.mat'];
save_file_dir = ['seg_res/', dataset, '/'];

load(img_file_path, 'img');
ori_gray_img = double(img);
[height, width] = size(img);
if ~exist(save_file_dir, 'dir')
    mkdir(save_file_dir);
end

%% num superpixel
% IP
% n_sp=50:25:550;

% PU
% n_sp=100:100:2000;

% TR
% n_sp=2900:150:5900;

% BO
n_sp=2900:150:5900;

for i=n_sp
    img = ori_gray_img;
    seg_res = mex_ers(img, i);
    seg_res = int16(seg_res);
    save_file_path=[save_file_dir, dataset, '_sp_map_', num2str(i), '.mat'];
    save(save_file_path,'seg_res');
end