% =========================================================================
% 训练模型
% =========================================================================

clear all; close all; clc;

% -------------------------------------------------------------------------
% parameter setting
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 5;                            % number of neighbors for local coding
c = 10;                             % regularization parameter for linear SVM
                                    % in Liblinear package 
% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package, you need 
                                    % download and compile the matlab codes

img_dir = 'image/flower3/train';            % directory for the image database                             
data_dir = 'data/sift/flower3/train';       % directory for saving SIFT descriptors
fea_dir = 'features/flower3/train';    % directory for saving final image features

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
    nCodebook = size(B, 2); % size of the codebook  nCodebook为矩阵B的列数1024

    % ---------------------------------------------------------------------
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
    nCodebook = size(B, 2); % size of the codebook  nCodebook为矩阵B的列数1024
    
    % ---------------------------------------------------------------------
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

fprintf('\nTraining...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);

tr_idx = [];
    
for jj = 1:nclass,
	idx_label = find(fdatabase.label == clabel(jj));   
    num = length(idx_label);
    tr_idx = [tr_idx; idx_label(1:num)];
end
    
fprintf('Training number: %d\n', length(tr_idx));
    
% load the training features 
tr_fea = zeros(length(tr_idx), dFea);
tr_label = zeros(length(tr_idx), 1);
    
for jj = 1:length(tr_idx),
    fpath = fdatabase.path{tr_idx(jj)};
    load(fpath, 'fea', 'label');
    tr_fea(jj, :) = fea';
    tr_label(jj) = label;
end

options = ['-c ' num2str(c)];
model = train(double(tr_label), sparse(tr_fea), options);
save model model;
% -------------------------------------------------------------------------
    
