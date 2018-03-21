%used in learn_dict and upscale method
%  many times in scaleup_GR
% return features for some imgs
function [features] = collect(conf, imgs, scale, filters, verbose)

if nargin < 5
    verbose = 0;
end

num_of_imgs = numel(imgs);
feature_cell = cell(num_of_imgs, 1); % contains images' features
num_of_features = 0;

if verbose
    fprintf('Collecting features from %d image(s) ', num_of_imgs)
end
feature_size = [];

h = [];

%% Collecting features into feature_cell £¨cell£©
for i = 1:num_of_imgs
    h = progress(h, i / num_of_imgs, verbose);      %what does var 'h' used for ???
    sz = size(imgs{i});
    if verbose
        fprintf(' [%d x %d]', sz(1), sz(2));
    end

    F = extract(conf, imgs{i}, scale, filters);              %extract features
    num_of_features = num_of_features + size(F, 2);     %nums of features for each img are different ???
    feature_cell{i} = F;

    assert(isempty(feature_size) || feature_size == size(F, 1), ...
        'Inconsistent feature size!')  %feature_size changed in extract ???
    feature_size = size(F, 1);            %in extract : feature_size = prod(conf.window) * numel(conf.filters);
end
if verbose
    fprintf('\nExtracted %d features (size: %d)\n', num_of_features, feature_size);
end

%% store into features (matrix)
clear imgs % to save memory
% features are enough, and the origin imgs are not used any more.
features = zeros([feature_size num_of_features], 'single');
offset = 0;
for i = 1:num_of_imgs
    F = feature_cell{i};
    N = size(F, 2); % number of patches in current img
    features(:, (1:N) + offset) = F;
    offset = offset + N;
end
