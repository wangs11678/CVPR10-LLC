function [database] = CalculateLbpDescriptor(rt_img_dir, rt_data_dir, maxImSize)
%==========================================================================
% usage: calculate the hog descriptors given the image directory
%
% inputs
% rt_img_dir    -image database root path
% rt_data_dir   -feature database root path
% maxImSize     -maximum size of the input image
%
% outputs
% database      -directory for the calculated sift features
%
%==========================================================================

disp('Extracting LBP features...');
subfolders = dir(rt_img_dir);


database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_img_dir, subname, '*.jpg'));
        
        c_num = length(frames);           
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        lbppath = fullfile(rt_data_dir, subname);        
        if ~isdir(lbppath),
            mkdir(lbppath);
        end;
        
        for jj = 1:c_num,
            imgpath = fullfile(rt_img_dir, subname, frames(jj).name);
            
            I = imread(imgpath);
            if ndims(I) == 3,
                I = im2double(rgb2gray(I));
            else
                I = im2double(I);
            end;
            
            [im_h, im_w] = size(I);
            
            if max(im_h, im_w) > maxImSize,
                I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
                [im_h, im_w] = size(I);
            end;

            % find LBP descriptors
            mapping=getmapping(8,'u2');
            lbpArr=lbp(I,1,8,mapping,'h'); %LBP histogram in (8,1) neighborhood using uniform patterns
            
            fea = lbpArr';
            
            [pdir, fname] = fileparts(frames(jj).name);                        
            fpath = fullfile(rt_data_dir, subname, [fname, '.mat']);
            
            save(fpath, 'fea');
            database.path = [database.path, fpath];
        end;    
    end;
end;
