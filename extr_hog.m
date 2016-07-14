function extr_hog(img_dir, data_dir)
% for example
% img_dir = 'image/Caltech101';
% data_dir = 'data/Caltech101';

addpath('hog');

maxImSize = 300;

[database] = CalculateHogDescriptor(img_dir, data_dir, maxImSize);