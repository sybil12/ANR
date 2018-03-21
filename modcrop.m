% used in go_run_upscaling_experiment.m and learn_dict
%  cut  redundant parts of imgs
% imgs is a cell, which contains some images
% modulo?
function imgs = modcrop(imgs, modulo)

for i = 1:numel(imgs)
    sz = size(imgs{i});
    sz = sz - mod(sz, modulo);  %mod ШЁгр
     imgs{i} = imgs{i}(1:sz(1), 1:sz(2));
end
