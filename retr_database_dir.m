function [database] = retr_database_dir(rt_data_dir)
%=========================================================================
% inputs
% rt_data_dir   -the rootpath for the database. e.g. '../data/caltech101'
% outputs
% database      -a tructure of the dir
%                   .path   pathes for each image file
%                   .label  label for each image file
% written by Jianchao Yang
% Mar. 2009, IFP, UIUC
%=========================================================================

fprintf('dir the database...');
subfolders = dir(rt_data_dir);

database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

fprintf('\n');
for ii = 1:length(subfolders),
    subname = subfolders(ii).name;
    %fprintf('subfolders: %s\n', subname); %显示data\Caltech101中子文件夹（各类）名字
    
    if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
        database.nclass = database.nclass + 1;
        
        database.cname{database.nclass} = subname;
        
        frames = dir(fullfile(rt_data_dir, subname, '*.mat'));
        c_num = length(frames);
        %fprintf('frames: %d\n', c_num); %显示data\Caltech101中子文件夹（各类）数量
                    
        database.imnum = database.imnum + c_num;
        database.label = [database.label; ones(c_num, 1)*database.nclass];
        
        for jj = 1:c_num,
            c_path = fullfile(rt_data_dir, subname, frames(jj).name);
            %fprintf('c_path: %s\n', c_path); %显示data\Caltech101中子文件夹（各类）中各个图片的路径
            database.path = [database.path, c_path];
        end;    
    end;
end;
disp('done!');