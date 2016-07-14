function extr_lbp(img_dir, data_dir)
% for example
% img_dir = 'image/Caltech101';
% data_dir = 'data/Caltech101';

addpath('lbp');

maxImSize = 300;

[database] = CalculateLbpDescriptor(img_dir, data_dir, maxImSize);