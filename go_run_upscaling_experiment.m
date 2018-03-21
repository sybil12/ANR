% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution
% Example code
%
% March 22, 2013. Radu Timofte, VISICS @ KU Leuven
%
% Revised version:
% October 3, 2013. Radu Timofte, CVL @ ETH Zurich
%
% Reference:
% Radu Timofte, Vincent De Smet, Luc Van Gool.
% Anchored Neighborhood Regression for Fast Example-Based Super-Resolution.
% International Conference on Computer Vision (ICCV), 2013.
%
% For any questions, email me by timofter@vision.ee.ethz.ch
%


% conf.scale = upscaling; % scale-up factor
% conf.upsample_factor = upscaling; % upsample low-res. into mid-res.

clear;

p = pwd;
addpath(fullfile(p, '/methods'));  % the upscaling methods

%       % make sure you have make ompbox and ksvdbox already,
%       % if not run the code, and before make you should have a gcc compiler
% addpath(fullfile(p, '/ksvdbox')) % K-SVD dictionary training algorithm
% addpath(fullfile(p, '/ompbox')) % Orthogonal Matching Pursuit algorithm

imgscale = 1; % the scale reference we work with
flag = 0;       % flag = 0 - only GR, ANR and bicubic methods, the other get the bicubic result by default
                % flag = 1 - all the methods are applied

upscaling = 3; % the magnification factor x2, x3, x4...

input_dir = 'Set5'; % Directory with input images from Set5 image dataset
%input_dir = 'Set14'; % Directory with input images from Set14 image dataset

pattern = '*.bmp'; % Pattern to process

dict_sizes = [16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536];
neighbors = [1:1:12, 16:4:32, 40:8:64, 80:16:128, 256, 512, 1024];
%d = 7
%for nn=1:28
%nn= 28

disp(['The experiment uses ' input_dir ' dataset and aims at a magnification of factor x' num2str(upscaling)]);
if flag==1
    disp('All methods are employed : Bicubic, Yang et al., Zeyde et al., GR, ANR, NE+LS, NE+NNLS, NE+LLE');
else
    disp('We run only for Bicubic, GR and ANR methods, the other get the Bicubic result by default');
end

fprintf('\n\n');

for d=7  %choose the dict scale from dict_sizes
    tag = [input_dir '_x' num2str(upscaling) '_' num2str(dict_sizes(d)) 'atoms'];

    disp(['Upscaling x' num2str(upscaling) ' ' input_dir ' with Zeyde dictionary of size = ' num2str(dict_sizes(d))]);

    mat_file = ['conf_Zeyde_' num2str(dict_sizes(d)) '_finalx' num2str(upscaling)];

    % dict
    if exist([mat_file '.mat'],'file')
        disp(['Load trained dictionary...' mat_file]);
        load(mat_file, 'conf');
    else     %trian hr and lr dict
        disp(['Training dictionary of size ' num2str(dict_sizes(d)) ' using Zeyde approach...']);
        % Simulation settings
        conf.scale = upscaling; % scale-up factor
        conf.level = 1; % # of scale-ups to perform ????
        conf.window = [3 3]; % low-res. window size
        conf.border = [1 1]; % border of the image (to ignore)

        % High-pass filters for feature extraction (defined for upsampled low-res.)
        conf.upsample_factor = upscaling; % upsample low-res. into mid-res.
        O = zeros(1, conf.upsample_factor-1);
        G = [1 O -1]; % Gradient
        L = [1 O -2 O 1]/2; % Laplacian
        conf.filters = {G, G.', L, L.'}; % 2D versions
        conf.interpolate_kernel = 'bicubic';

        conf.overlap = [1 1]; % partial overlap (for faster training)
        if upscaling <= 2   % why ???
            conf.overlap = [2 2]; % partial overlap (for faster training)
        end

        startt = tic;
        conf = learn_dict(conf, ....
            load_images( glob('CVPR08-SR/Data/Training', '*.bmp')  ),...
            dict_sizes(d));
        conf.overlap = conf.window - [1 1]; % full overlap scheme (for better reconstruction)
        conf.trainingtime = toc(startt);
        toc(startt)

        save(mat_file, 'conf');

        % train call
    end

    % set lambda
    if dict_sizes(d) < 1024
        lambda = 0.01;
    elseif dict_sizes(d) < 2048
        lambda = 0.1;
    elseif dict_sizes(d) < 8192
        lambda = 1;
    else
        lambda = 5;
    end

    % count ProjM and PP
    if dict_sizes(d) < 10000
        %conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))*conf.dict_lores';
        conf.ProjM = (conf.dict_lores'*conf.dict_lores+lambda*eye(size(conf.dict_lores,2)))\conf.dict_lores';

        conf.PP = (1+lambda)*conf.dict_hires*conf.ProjM;
    else
        % here should be an approximation
        conf.PP = zeros(size(conf.dict_hires,1), size(conf.V_pca,2));
        conf.ProjM = [];
    end

    % get all img names from the dir (Cell array)
    conf.filenames = glob(input_dir, pattern);
    %conf.filenames = {conf.filenames{4}};

    %method name
    conf.desc = {'Original', 'Bicubic', 'Yang et. al.', ...
        'Zeyde et. al.', 'Our GR', 'Our ANR', ...
        'NE+LS','NE+NNLS','NE+LLE'};
    conf.results = {};


    %conf.points = 1:10:size(conf.dict_lores,2);    % index of atoms
    conf.points = 1:1:size(conf.dict_lores,2);

    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';


    % precompute for ANR the anchored neighborhoods and
    % the projection matrices for the dictionary

    %% count PPs , project matrix when uses K neighbours for each atom
    conf.PPs = [];
    % clustersz , the K in paper , num of neighbours
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(conf.pointslo'*conf.dict_lores);       % Correlation matrix
    %D = conf.pointslo'*conf.dict_lores;

    % each cloumn in PPs is correspond to a atom in D_l
    for i = 1:length(conf.points)
        [~,idx] = sort(D(i,:), 'descend');  % idx represent the origin index in i row of D
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));
            %conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))/(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';
        end
    end

    save([tag '_' mat_file '_ANR_projections_imgscale_' num2str(imgscale)],'conf'); %save current var -- conf

    %%
    conf.result_dirImages = qmkdir([input_dir '/results_' tag]);
    conf.result_dirImagesRGB = qmkdir([input_dir '/results_' tag 'RGB']);
    conf.result_dir = qmkdir(['Results-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);
    conf.result_dirRGB = qmkdir(['ResultsRGB-' datestr(now, 'YYYY-mm-dd_HH-MM-SS')]);

    t = cputime;

    conf.countedtime = zeros(numel(conf.desc),numel(conf.filenames)); %desc-- methods names; filenames-- imags names

    res =[];
    %% upscaling i_th image
    for i = 1:numel(conf.filenames)
        f = conf.filenames{i};
        [p, n, x] = fileparts(f);
        [img, imgCB, imgCR] = load_images({f});
        if imgscale<1  %when imgscale < 1???
            img = resize(img, imgscale, conf.interpolate_kernel);
            imgCB = resize(imgCB, imgscale, conf.interpolate_kernel);
            imgCR = resize(imgCR, imgscale, conf.interpolate_kernel);
        end
        sz = size(img{1});

        fprintf('%d/%d\t"%s" [%d x %d]\n', i, numel(conf.filenames), f, sz(1), sz(2));

        img = modcrop(img, conf.scale^conf.level);
        imgCB = modcrop(imgCB, conf.scale^conf.level);
        imgCR = modcrop(imgCR, conf.scale^conf.level);
            %down sampling
            low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);
            if ~isempty(imgCB{1})
                lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
                lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
            end
        %cheap upscaling
        interpolated = resize(low, conf.scale^conf.level, conf.interpolate_kernel);
        if ~isempty(imgCB{1})
            interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);
            interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);
        end

        res{1} = interpolated;

        if (flag == 1) && (dict_sizes(d) == 1024)
            startt = tic;
            res{2} = {yima(low{1}, upscaling)};
            toc(startt)
            conf.countedtime(2,i) = toc(startt);
        else
            res{2} = interpolated;
        end

        if (flag == 1)
            startt = tic;
            res{3} = scaleup_Zeyde(conf, low);
            toc(startt)
            conf.countedtime(3,i) = toc(startt);
        else
            res{3} = interpolated;
        end

        %if flag == 1
            startt = tic;
            res{4} = scaleup_GR(conf, low);
            toc(startt)
            conf.countedtime(4,i) = toc(startt);
        %else
            %res{4} = interpolated;
        %end

        startt = tic;
        res{5} = scaleup_ANR(conf, low);
        toc(startt)
        conf.countedtime(5,i) = toc(startt);

        if flag == 1
            startt = tic;
            if 12 < dict_sizes(d)
                res{6} = scaleup_NE_LS(conf, low, 12);
            else
                res{6} = scaleup_NE_LS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(6,i) = toc(startt);
        else
            res{6} = interpolated;
        end

        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{7} = scaleup_NE_NNLS(conf, low, 24);
            else
                res{7} = scaleup_NE_NNLS(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(7,i) = toc(startt);
        else
            res{7} = interpolated;
        end

        if flag == 1
            startt = tic;
            if 24 < dict_sizes(d)
                res{8} = scaleup_NE_LLE(conf, low, 24);
            else
                res{8} = scaleup_NE_LLE(conf, low, dict_sizes(d));
            end
            toc(startt)
            conf.countedtime(8,i) = toc(startt);
        else
            res{8} = interpolated;
        end

        result = cat(3, img{1}, interpolated{1}, res{2}{1}, res{3}{1}, ...
            res{4}{1}, res{5}{1}, res{6}{1}, res{7}{1}, res{8}{1});
        result = shave(uint8(result * 255), conf.border * conf.scale);

        if ~isempty(imgCB{1})
            resultCB = interpolatedCB{1};
            resultCR = interpolatedCR{1};
            resultCB = shave(uint8(resultCB * 255), conf.border * conf.scale);
            resultCR = shave(uint8(resultCR * 255), conf.border * conf.scale);
        end

        conf.results{i} = {};
        for j = 1:numel(conf.desc)
            conf.results{i}{j} = fullfile(conf.result_dirImages, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            imwrite(result(:, :, j), conf.results{i}{j});

            conf.resultsRGB{i}{j} = fullfile(conf.result_dirImagesRGB, [n sprintf('[%d-%s]', j, conf.desc{j}) x]);
            if ~isempty(imgCB{1})
                rgbImg = cat(3,result(:,:,j),resultCB,resultCR);
                rgbImg = ycbcr2rgb(rgbImg);
            else
                rgbImg = cat(3,result(:,:,j),result(:,:,j),result(:,:,j));
            end

            imwrite(rgbImg, conf.resultsRGB{i}{j});
        end
        conf.filenames{i} = f;
    end

   %%
    conf.duration = cputime - t;  %time of duration for upscaling step

    % Test performance
    scores = run_comparison(conf);
    process_scores_Tex(conf, scores,length(conf.filenames));

    %run_comparisonRGB(conf); % provides color images and HTML summary

    save([tag '_' mat_file '_results_imgscale_' num2str(imgscale)],'conf');


end
%