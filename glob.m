%used in go_run_upscaling_experiment.m  and in load_images for learn_dict
% get all imgs from the dir
function result = glob(directory, pattern)

d = fullfile(directory, pattern);
files = dir(d);

result = cell(numel(files), 1);
for i = 1:numel(result)
    result{i} = fullfile(directory, files(i).name);
end
