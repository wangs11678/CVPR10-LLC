% =========================================================================
% 例子：
% hog+svm进行分类
% =========================================================================
clear all;
close all; 
clc;

% -------------------------------------------------------------------------
% parameter setting
c = 10;                             % regularization parameter for linear SVM in Liblinear package                             
nRounds = 10;                       % number of random test on the dataset
tr_num  = 200;                       % training examples per category

% -------------------------------------------------------------------------
% set path
addpath('Liblinear/matlab');        % we use Liblinear package, you need download and compile the matlab codes
img_dir = 'image/flower3';         % directory for the image database                             
data_dir = 'data/hog/flower3';     % directory for saving HOG descriptors

% -------------------------------------------------------------------------
% extract HOG descriptors
exist_data_hog = true;
if(exist_data_hog)
    extr_hog('F:\project\matlab\CVPR10-LLC\image\flower3','F:\project\matlab\CVPR10-LLC\data\hog\flower3');
end
% -------------------------------------------------------------------------

database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end

% -------------------------------------------------------------------------
% extract image features
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
    
    load(fpath, 'hogfea');
    sc_fea(:, iter1) = hogfea;
    sc_label(iter1) = database.label(iter1);
    
    feaPath = fpath;
    
    fdatabase.label(iter1) = flabel;
    fdatabase.path{iter1} = feaPath;
end;

% -------------------------------------------------------------------------
% evaluate the performance of the image feature using linear SVM
% we used Liblinear package in this example code

fprintf('\nTesting...\n');
clabel = unique(fdatabase.label);
nclass = length(clabel);
accuracy = zeros(nRounds, 1);

for ii = 1:nRounds,
    fprintf('Round: %d...\n', ii);
    tr_idx = [];
    ts_idx = [];
    
    for jj = 1:nclass,
        idx_label = find(fdatabase.label == clabel(jj));
        num = length(idx_label);
        
        idx_rand = randperm(num);
        
        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
    end
    
    fprintf('Training number: %d\n', length(tr_idx));
    fprintf('Testing number:%d\n', length(ts_idx));
    
    tr_fea = sc_fea(:, tr_idx)';
    tr_label = sc_label(tr_idx)';
    
    ts_fea = sc_fea(:, ts_idx)';
    ts_label = sc_label(ts_idx)';
    
    options = ['-c ' num2str(c)];
    model = train(double(tr_label), sparse(tr_fea), options);
    [C] = predict(ts_label, sparse(ts_fea), model);
    
    % normalize the classification accuracy by averaging over different
    % classes
    acc = zeros(nclass, 1);

    for kk = 1 : nclass,
        c = clabel(kk);
        idx = find(ts_label == c);
        curr_pred_label = C(idx);
        curr_gnd_label = ts_label(idx);    
        acc(kk) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
    end

    accuracy(ii) = mean(acc); 
    fprintf('Classification accuracy for round %d: %f\n', ii, accuracy(ii));
end

Ravg = mean(accuracy);                  % average recognition rate
Rstd = std(accuracy);                   % standard deviation of the recognition rate

fprintf('===============================================');
fprintf('Average classification accuracy: %f\n', Ravg);
fprintf('Standard deviation: %f\n', Rstd);    
fprintf('===============================================');
    
