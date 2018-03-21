% used in go_run_upscaling_experiment.m
% shave --Cut image border,
% only used for 1 and 2 dim ,here is img{1} and interpolated{1}
function I = shave(I, border)
I = I(1+border(1):end-border(1), ...
      1+border(2):end-border(2), :, :);
