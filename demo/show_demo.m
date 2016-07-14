load test.mat
%j = randperm(660);

for i = 1:660
    path =['.\image' fdatabase.path{ts_idx(i)}(9:end-4) '.png'];
    img = imread(path);
    imshow(img);
    fprintf('pred:%d, groundtruth:%d (1:bud, 2:road, 3:rose)\n', C(i), ts_label(i));
    if C(i) ~= ts_label(i)
        pause(10);
    end
end