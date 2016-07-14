function [database] = CalculateHogDescriptor(rt_img_dir, rt_data_dir, maxImSize)
%==========================================================================
% usage: calculate the hog descriptors given the image directory
%
% inputs
% rt_img_dir    -image database root path
% rt_data_dir   -feature database root path
% maxImSize     -maximum size of the input image
%
% outputs
% database      -directory for the calculated hog features
%
%==========================================================================

disp('Extracting HOG features...');
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
        
        hogpath = fullfile(rt_data_dir, subname);        
        if ~isdir(hogpath),
            mkdir(hogpath);
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
                %I = imresize(300,300);
                [im_h, im_w] = size(I);
            end;

            % make grid sampling SIFT descriptors
%             remX = mod(im_w-patchSize,gridSpacing);
%             offsetX = floor(remX/2)+1;
%             remY = mod(im_h-patchSize,gridSpacing);
%             offsetY = floor(remY/2)+1;
%     
%             [gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);
%             
%             fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
%                      frames(jj).name, im_w, im_h, size(gridX, 2), size(gridX, 1), numel(gridX));
                 
            % find HOG descriptors
            hogArr = hog_feature_vector(I);
            
            fea = hogArr';
%             hogLens = [hogtLens; hoglen];
%             
%             feaSet.feaArr = hogArr';
%             feaSet.x = gridX(:) + patchSize/2 - 0.5;
%             feaSet.y = gridY(:) + patchSize/2 - 0.5;
%             feaSet.width = im_w;
%             feaSet.height = im_h;
            
            [pdir, fname] = fileparts(frames(jj).name);                        
            fpath = fullfile(rt_data_dir, subname, [fname, '.mat']);
            
            save(fpath, 'fea');
            database.path = [database.path, fpath];
        end;    
    end;
end;
% lenStat = hist(hogLens, 100);