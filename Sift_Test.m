% =========================================================================
% 测试结果
% =========================================================================

clear all; close all; clc;

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 5;                            % number of neighbors for local coding
c = 10;                             % regularization parameter for linear SVM in Liblinear package

mem_block = 3000;                   % maxmum number of testing features loaded each time  


% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package, you need 
                                    % download and compile the matlab codes

img_dir = 'image/flower3/test';            % directory for the image database                             
data_dir = 'data/sift/flower3/test';       % directory for saving SIFT descriptors
fea_dir = 'features/flower3/test';    % directory for saving final image features


% -------------------------------------------------------------------------
% extract SIFT descriptors, we use Prof. Lazebnik's matlab codes in this package
% change the parameters for SIFT extraction inside function 'extr_sift'
% extr_sift(img_dir, data_dir);

noexist_data_sift = true;
%如果data不存在，提取sift描述子
if(noexist_data_sift)
    extr_sift(img_dir,data_dir);
end
% -------------------------------------------------------------------------

noexist_fea_sift = true;
%如果features不存在，load data进行coding产生fea
if(noexist_fea_sift)  
    % retrieve the directory of the database and load the codebook
    database = retr_database_dir(data_dir);
    if isempty(database),
        error('Data directory error!');
    end
    Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];
    load(Bpath);
    nCodebook = size(B, 2);              % size of the codebook  nCodebook为矩阵B的列数1024

    % -------------------------------------------------------------------------
    % extract image features
    dFea = sum(nCodebook*pyramid.^2);
    nFea = length(database.path);
    fdatabase = struct;
    fdatabase.path = cell(nFea, 1);         % path for each image feature
    fdatabase.label = zeros(nFea, 1);       % class label for each image feature

    for iter1 = 1:nFea,  
        if ~mod(iter1, 5),
           fprintf('.');
        end
        if ~mod(iter1, 100),
            fprintf(' %d images processed\n', iter1);
        end
        fpath = database.path{iter1};
        flabel = database.label(iter1);

        load(fpath);
        [rtpath, fname] = fileparts(fpath);
        feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);

        fea = LLC_pooling(feaSet, B, pyramid, knn);
        label = database.label(iter1);
        
        if ~isdir(fullfile(fea_dir, num2str(flabel))),
            mkdir(fullfile(fea_dir, num2str(flabel)));
        end      
        save(feaPath, 'fea', 'label');
        fdatabase.label(iter1) = flabel;
        fdatabase.path{iter1} = feaPath;
    end;
%如果features已经存在，直接load fea
else 
    fdatabase = retr_database_dir(fea_dir);
    if isempty(fdatabase),
        error('Data directory error!');
    end
    Bpath = ['dictionary/Caltech101_SIFT_Kmeans_1024.mat'];
    load(Bpath);
    nCodebook = size(B, 2);              % size of the codebook  nCodebook为矩阵B的列数1024
    
    % -------------------------------------------------------------------------
    dFea = sum(nCodebook*pyramid.^2);
    nFea = length(fdatabase.path);
    for iter1 = 1:nFea,   
        feaPath = fdatabase.path{iter1};
        load(feaPath);
        label = fdatabase.label(iter1);
    end
end

% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

fprintf('\nTesting...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);

ts_idx = []; 
for jj = 1:nclass,
    idx_label = find(fdatabase.label == clabel(jj));
    num = length(idx_label);
    ts_idx = [ts_idx; idx_label(1:num)];
end

fprintf('Testing number:%d\n', length(ts_idx));
    
% -------------------------------------------------------------------------
% load model
load('model.mat')
% -------------------------------------------------------------------------

% load the testing features
ts_num = length(ts_idx);
ts_label = [];
    
if ts_num < mem_block,
    % load the testing features directly into memory for testing
    ts_fea = zeros(length(ts_idx), dFea);
    ts_label = zeros(length(ts_idx), 1);

    for jj = 1:length(ts_idx),
        fpath = fdatabase.path{ts_idx(jj)};
        load(fpath, 'fea', 'label');
        ts_fea(jj, :) = fea';
        ts_label(jj) = label;
    end

    [C] = predict(ts_label, sparse(ts_fea), model);
else
    % load the testing features block by block
    num_block = floor(ts_num/mem_block);
    rem_fea = rem(ts_num, mem_block);
        
    curr_ts_fea = zeros(mem_block, dFea);
    curr_ts_label = zeros(mem_block, 1);
        
    C = [];
        
    for jj = 1:num_block,
        block_idx = (jj-1)*mem_block + (1:mem_block);
        curr_idx = ts_idx(block_idx); 
            
        % load the current block of features
        for kk = 1:mem_block,
            fpath = fdatabase.path{curr_idx(kk)};
            load(fpath, 'fea', 'label');
            curr_ts_fea(kk, :) = fea';
            curr_ts_label(kk) = label;
        end    
            
        % test the current block features
        ts_label = [ts_label; curr_ts_label];
        [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model);
        C = [C; curr_C];
    end
        
    curr_ts_fea = zeros(rem_fea, dFea);
    curr_ts_label = zeros(rem_fea, 1);
    curr_idx = ts_idx(num_block*mem_block + (1:rem_fea));
        
    for kk = 1:rem_fea,
       fpath = fdatabase.path{curr_idx(kk)};
       load(fpath, 'fea', 'label');
       curr_ts_fea(kk, :) = fea';
       curr_ts_label(kk) = label;
    end  
        
    ts_label = [ts_label; curr_ts_label];
    [curr_C] = predict(curr_ts_label, sparse(curr_ts_fea), model); 
    C = [C; curr_C];        
end

% -------------------------------------------------------------------------
% 展示结果
data = retr_database_img_dir(img_dir);
for i = 1:ts_num
    %path =['.\image' fdatabase.path{ts_idx(i)}(9:end-4) '.jpg'];    path = data.path{i};
    %img = imread(data.path{i});
    %imshow(img);
    fprintf('groundtruth:%d, pred:%d\n', ts_label(i), C(i));
    %if C(i) ~= ts_label(i)
        %pause(10);
    %end
end
% -------------------------------------------------------------------------

% normalize the classification accuracy by averaging over different classes

acc = zeros(nclass, 1);
predright = 0;
sum = 0;
for jj = 1 : nclass,
    c = clabel(jj);
    idx = find(ts_label == c);
    curr_pred_label = C(idx);
    curr_gnd_label = ts_label(idx);    
    acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    fprintf('第%d类准确率：%f%%\n', jj, 100*acc(jj));
    predright = predright + length(find(curr_pred_label == curr_gnd_label));
    sum = sum + length(idx);
end

fprintf('===============================================\n');
fprintf('Accuracy: %f%%\n', 100*double(predright/sum)); 
fprintf('===============================================\n');

% -------------------------------------------------------------------------    
