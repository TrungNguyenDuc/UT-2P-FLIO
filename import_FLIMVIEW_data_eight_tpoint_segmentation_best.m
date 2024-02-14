addpath('scripts');
addpath('./functions');
clear all;
close all;

%% all setting
is_adjust = 1; %calib the phasor plot
is_peak = 1; %calib with decay calculate from peak location
lifetime_limit = [0 3]; % for visualization only, still keep the data without filtering
nonlinear_cmap_setting = [0.1 3]; % for visualization only, with non linear color code
cluster_number = 3; % cluster number 3-5
isfilter = 1; % apply median filter or not on G_t, S_t separately
medfilt_size = 3; % median filter size default 3/ extreme 5
ntimes = 1; % median filter repeat
ITrel = 0.1; % relative percentage of intensity as lower threshold
outliers_remove = 1; % remove outlier in boxplot and nhist

model.top_left= 0;             % # of data points on the left of the peak. 5 is also fine- Important
model.top_right= 70;          % # of data points on the right of the peak

%%
[file_collect,path] = uigetfile('*.tif','MultiSelect','on');
if isequal(file_collect,0)
    disp('User selected Cancel');
else
    disp(['User selected ', fullfile(path,file_collect)]);
end
if class(file_collect) == 'char'
    number_of_timepoints = 1;
else
    number_of_timepoints = length(file_collect);
end
cluster_results = cell(number_of_timepoints,6);
cluster_segmented_photoreceptor = cell(number_of_timepoints,8);
cluster_segmented_RPE = cell(number_of_timepoints,8);
%%
for timepoint = 1:number_of_timepoints
    if number_of_timepoints == 1
        file = file_collect;
    else
        file = file_collect{timepoint};
    end

    %file = file_collect{timepoint};
    file_temp = fullfile(path,file);
    info = imfinfo(file_temp);
    num_images = numel(info) ;
    image_tif_matrix = zeros(info(1).Width,info(1).Height, num_images);

    for jj = 1:num_images
        image_tif_matrix(:,:,jj) = double(imread(file_temp, jj, 'Info', info));
    end

    image_tif_matrix_size = size(image_tif_matrix);
    image_size = image_tif_matrix_size(1);
    bin_size = image_tif_matrix_size(3);

    file = file(1:end-8);


    % total_intensity
    total_intensity = sum(image_tif_matrix,3);
    total_intensity = reshape(total_intensity, [image_size image_size]);
    total_intensity_fig = figure();
    imagesc(total_intensity)
    axis square;
    %colormap gray;
    colorbar;
    axis off

  


    % show total decay curves
    total_decay = sum(image_tif_matrix,1);
    total_decay = sum(total_decay,2);
    total_decay = reshape(total_decay,[1 bin_size]);
    figure()
    plot(total_decay)

    %
  
    total_decay = image_tif_matrix(105,136,:)
    total_decay = reshape(total_decay,[1 bin_size]);
    figure()
    plot(total_decay)

    %
    %     pixel = [137 202];
    %     figure()
    %     plot(reshape(image_tif_matrix(pixel(1),pixel(2),:),[1 bin_size]))



    model.top_left= 0;             % # of data points on the left of the peak. 5 is also fine- Important
    model.top_right= 70;          % # of data points on the right of the peak
    %intensity threshold set at 3000
    %[mask_combine,G_t,S_t,total_top_tiff,fig_phasor_plot,n_time_bin,omega]=visualize_filter_lifetime_phasor_FLIO(image_tif_matrix,'green',model.top_left,model.top_right,image_size,path,file);
    %[G_t,S_t,total_top_tiff,n_time_bin,omega,top_matrix_gauss]= visualize_filter_lifetime_phasor_FLIO_without_hand_select(image_tif_matrix,'green',model.top_left,model.top_right,image_size,path,file,is_adjust);
    [G_t,S_t,total_top_tiff,n_time_bin,omega,top_matrix_gauss]=visualize_filter_lifetime_phasor_FLIO_without_hand_select_peak(image_tif_matrix,'green',model.top_left,model.top_right,image_size,path,file,is_adjust,is_peak);
    %Kmeans_unmixing(G_t,S_t,total_intensity,4,1,10,true,1,n_time_bin)


    %     lifetime_limit = [0 3];
    %     nonlinear_cmap_setting = [0.1 3];
    %     cluster_number = 3;
    %     isfilter = 0;
    %     medfilt_size = 3;
    %     ntimes = 1;
    %     ITrel = 0.1;
    [cluster_prob,cluster_intensity,cluster_filtered_phasor_lifetime,cluster_filtered_phasor_lifetime_without_limits]=GMM_unmixing(G_t,S_t,total_top_tiff,cluster_number,isfilter,medfilt_size,ntimes,ITrel,1,n_time_bin,path,'green',omega,lifetime_limit,nonlinear_cmap_setting,timepoint);

   % [cluster_prob,cluster_intensity,cluster_filtered_phasor_lifetime,cluster_filtered_phasor_lifetime_without_limits]=GMM_unmixing(G_t,S_t,total_top_tiff,3,isfilter,medfilt_size,ntimes,ITrel,1,n_time_bin,path,'green',omega,lifetime_limit,nonlinear_cmap_setting,timepoint);

    cluster_results{timepoint,1} = cluster_prob;
    cluster_results{timepoint,2} = cluster_intensity;
    cluster_results{timepoint,3} = cluster_filtered_phasor_lifetime;
    cluster_results{timepoint,4} = cluster_filtered_phasor_lifetime_without_limits;
    cluster_results{timepoint,5} = G_t;
    cluster_results{timepoint,6} = S_t;

end
%%
figure();
%imagesc(cluster_results{7,2}{1})
% for timepoint = 1:number_of_timepoints
%     figure();
%     imagesc(cluster_results{timepoint,3}{k(timepoint)})
% end
%%
number_of_timepoints = 6
close all
%FOV 1 
%k = [1 1 1 1 1 1 1 1];
%FOV 2 
% k = [2 2 2 2 1 2 2]; % cluster number
%DB 1
%k = [2 2 1 2 2 2 2]
%DB 2
%k = [1 2 2 2 2 2 2];
%DB new FOV4
%k = [1 1 1 2 2 2 1]
%DB new FOV1
%k = [1 1 1 1 1 1 1 1]
%DB old FOV3
k = [1 1 1 1 2 1 1 1]
for timepoint = 1:number_of_timepoints
    figure();
    imagesc(cluster_results{timepoint,2}{k(timepoint)}) 
    % timepoint,2 intensity can change to timepoint,3 -> filter lifetime
    % Photoreceptor segmentation
    [L, num] = photoreceptor_segmentation_func(cluster_results{timepoint,2}{k(timepoint)});
    cluster_segmented_photoreceptor{timepoint,1} = L;
    cluster_segmented_photoreceptor{timepoint,2} = num;
    % cluster numbered
    %0 is border
    %1 is back ground
    % figure();
    % imagesc(L);
    % axis square;

    % get average value for each cluster
    % figure();
    cluster_lifetime = cluster_results{timepoint,4}{k(timepoint)};
    cluster_g = cluster_results{timepoint,5}(2:255,2:255);
    cluster_s = cluster_results{timepoint,6}(2:255,2:255);


    % imagesc(cluster_lifetime);
    % axis square;
    cluster_mean_lifetime = zeros(1,num);
    cluster_mean_g = zeros(1,num);
    cluster_mean_s = zeros(1,num);
    cluster_mean_lifetime_image = zeros(size(cluster_lifetime));
    cluster_decays = cell(1,num);

    for i = 1:num % start from 0 or 1
        %[row,col] = find(a==8)
        cluster_mask = L;
        cluster_mask(cluster_mask ~=i) = 0;
        cluster_mask(cluster_mask ==i) = 1;
        cluster_pixel_num = sum(cluster_mask(:) == 1);
        cluster_lifetime_masked = cluster_lifetime.*cluster_mask;
        cluster_mean_lifetime(i) = sum(cluster_lifetime_masked,'all')/cluster_pixel_num;
        cluster_mean_lifetime_image = cluster_mean_lifetime_image + (cluster_mask)*cluster_mean_lifetime(i);

        cluster_g_masked = cluster_g.*cluster_mask;
        cluster_mean_g(i) = sum(cluster_g_masked,'all')/cluster_pixel_num;

        cluster_s_masked = cluster_s.*cluster_mask;
        cluster_mean_s(i) = sum(cluster_s_masked,'all')/cluster_pixel_num;
        %cluster_segmented_photoreceptor{timepoint,3} = cluster_mean_lifetime;
        % figure();
        % imagesc(cluster_lifetime_masked)
        matrix_temp =top_matrix_gauss(2:255,2:255,:);
        single_element_matrix = bsxfun(@times, matrix_temp, cast(cluster_mask, 'like', matrix_temp));
        cluster_decays{i} = reshape(sum(single_element_matrix,[1 2]),1,[]);
        %cluster_decays{i} = calculate_decay(single_element_matrix);
    end
    %
    cluster_segmented_photoreceptor{timepoint,3} = cluster_mean_lifetime;
    cluster_segmented_photoreceptor{timepoint,4} = cluster_mean_lifetime_image;
    cluster_segmented_photoreceptor{timepoint,5} = cluster_decays;
    cluster_segmented_photoreceptor{timepoint,6} = cluster_mean_g;
    cluster_segmented_photoreceptor{timepoint,7} = cluster_mean_s;

    lifetime_limit = [1 3];
    figure();
    histogram(cluster_mean_lifetime(2:end),20,'BinLimits',lifetime_limit);
    figure()
    boxplot(cluster_mean_lifetime(2:end))
end

%%
% for i = 1:timepoint
% figure()
% imagesc(cluster_segmented_photoreceptor{timepoint,4})
% axis square
% end
%% box plot for segmentation
number_of_timepoints = 6
lifetime_threshold1 = 1;
lifetime_threshold2 = 5;
hist_summary =[];
hist_remap = [];
hist_mean = [];
binWidth = 0.1;
hcounts = cell(number_of_timepoints,2); 



%number_of_timepoints = 4;
% a = cluster_segmented_photoreceptor{1,3}
% cluster_segmented_photoreceptor{1,3} = cluster_segmented_photoreceptor{2,3}
% cluster_segmented_photoreceptor{2,3} = a
for i = 1:number_of_timepoints
    temp = cluster_segmented_photoreceptor{i,3}((cluster_segmented_photoreceptor{i,3}>lifetime_threshold1) & (cluster_segmented_photoreceptor{i,3}<lifetime_threshold2))';
    [hcounts{i,1}, hcounts{i,2}] = histcounts(temp,'BinWidth',binWidth); 
    hist_summary= cat(1,hist_summary,temp);
    hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
    hist_mean = cat(1,hist_mean,mean(temp));
end
maxCount = max([hcounts{:,1}]);
figure();
clf
boxplot(hist_summary,hist_remap)
hold on
plot(hist_mean, '*b')
hold off
%%

load('excitation_scheme.mat');
figure();
clf
%pos = [1, 10, 30, 40, 60, 70];
pos = [2.5, 12.5, 32.5, 42.5, 62.5, 72.5];
%pos = [2.5, 12.5, 32.5, 42.5, 62.5, 72.5, 92.5, 102.5];
yyaxis left
colors = lines(length(pos)); 
boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
hold on
scatter(pos,hist_mean(1:6),50,colors,"filled")
plot(pos,hist_mean, 'LineWidth', 1,'Color','r','LineStyle','- -');

xInterval = 1; %x-interval is always 1 with boxplot groups
%normwidth = (1-hgapGrp-hgap)/2;    
binWidth = 0.01;  % histogram bin widths
hgapGrp = .15;   % horizontal gap between pairs of boxplot/histograms (normalized)
hgap = 0.06;      % horizontal gap between boxplot and hist (normalized)
xCoordinate = pos;  %x-positions is always 1:n with boxplot groups
histX0 = xCoordinate + 5/2;    % histogram base
maxHeight = 4;       % max histogram height
patchHandles = gobjects(1,length(pos)); 
for i = 1:length(pos)
    % Normalize heights 
    height = hcounts{i,1}/maxCount*maxHeight;
    % Compute x and y coordinates 
    xm = [zeros(1,numel(height)); repelem(height,2,1); zeros(2,numel(height))] + histX0(i);
    yidx = [0 0 1 1 0]' + (1:numel(height));
    ym = hcounts{i,2}(yidx);
    % Plot patches
    patchHandles(i) = patch(xm(:),ym(:),colors(i,:),'EdgeColor',colors(i,:),'LineWidth',1,'FaceAlpha',.45);
end
xlim([-5 85])
ylim([0 5])

% xlim([-5 125])
% ylim([0 5])

hold on

%figure();
%clf
%pos = [1, 10, 30, 40, 60, 70];
%colors = lines(length(pos)); 
% % yyaxis left
% % boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
% % hold on
% % scatter(pos,hist_mean,50,colors,"filled")
% % plot(pos,hist_mean, 'LineWidth', 1,'Color','r','LineStyle','--');
yyaxis right
plot(excitation_scheme(:,1), excitation_scheme(:,2),'Color','r','LineWidth',2)
ylim([0 10])
xticks(0:5:75);
xticklabels(0:5:75);
hold off

%%
%% Add vertical histograms (patches) with matching colors

%%
% lifetime_threshold1 = 0.5;
% lifetime_threshold2 = 3;
% hist_summary =[];
% hist_remap = [];
% hist_mean = [];
% %number_of_timepoints = 4;
% % a = cluster_segmented_photoreceptor{1,3}
% % cluster_segmented_photoreceptor{1,3} = cluster_segmented_photoreceptor{2,3}
% % cluster_segmented_photoreceptor{2,3} = a
% for i = 1:number_of_timepoints
%     temp = cluster_segmented_photoreceptor{i,3}((cluster_segmented_photoreceptor{i,3}>lifetime_threshold1) & (cluster_segmented_photoreceptor{i,3}<lifetime_threshold2))';
%     hist_summary= cat(1,hist_summary,temp);
%     hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
%     hist_mean = cat(1,hist_mean,mean(temp));
% end
figure();
clf
pos = [1, 10, 30, 40, 60, 70];
colors = lines(length(pos)); 
yyaxis left
boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
hold on
scatter(pos,hist_mean,50,colors,"filled")
plot(pos,hist_mean, 'LineWidth', 1,'Color','r');
yyaxis right
plot(excitation_scheme(:,1), excitation_scheme(:,2))
ylim([0 1.2])
xticks(0:5:75);
xticklabels(0:5:75);
hold off

%%
% figure();
% clf
% pos = [1, 10, 30, 40, 60, 70];
% colors = lines(length(pos));
% hold on
% t = tiledlayout(1,1);
% ax1 = axes(t);
% pos = [1, 10, 30, 40, 60, 70];
% colors = lines(length(pos)); 
% boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
% scatter(pos,hist_mean,50,colors,"filled")
% plot(pos,hist_mean, 'LineWidth', 1,'Color','r');
% ax1.XColor = 'r';
% ax1.YColor = 'r';
% 
% ax2 = axes(t);
% 
% ax2.XAxisLocation = 'top';
% ax2.YAxisLocation = 'right';
% ax2.Color = 'none';
% ax1.Box = 'off';
% ax2.Box = 'off';




%% nhist

nhist_summary = cell(number_of_timepoints,1);

for i = 1:number_of_timepoints
    
   temp = cluster_segmented_photoreceptor{i,3}((cluster_segmented_photoreceptor{i,3}>lifetime_threshold1) & (cluster_segmented_photoreceptor{i,3}<lifetime_threshold2))';
    nhist_summary{i} = temp;
end

% 'separate' to plot each set on its own axis, but with the same bounds
% 'binfactor' change the number of bins used, larger value =more bins
% 'samebins' force all bins to be the same for all plots
% 'legend' add a legend in the graph (default for structs)
% 'noerror' remove the mean and std plot from the graph
% 'median' add the median of the data to the graph
% 'text' return many details about each graph even if not plotted
nhist(nhist_summary,'pdf','median','newfig','separate');
nhist(nhist_summary,'pdf','median','newfig');
nhist(nhist_summary,'median','newfig','separate','samebins','noerror','binfactor',1);
nhist(nhist_summary,'median','newfig');

%%
for timepoint = 1:number_of_timepoints
    G_t = cluster_segmented_photoreceptor{timepoint,6}(2:end);
    S_t = cluster_segmented_photoreceptor{timepoint,7}(2:end);
    figure()
    omega = plot_PhasorCircle_with_reference_lifetime(1,71);
    scatter(G_t,S_t);

    %scatplot(G_t',S_t')
    hold on
    plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'r')
    axis equal
    lifetime_mean = (mean(S_t,'all')/mean(G_t,'all'))/omega;
    formatSpec = 'timepoint %i: G_t is %4.4f - S_t is %4.4f mm\n- mean lifetime is %4.4f ns\n';
    fprintf(formatSpec,timepoint,mean(G_t,'all'),mean(S_t,'all'),lifetime_mean)
end
%% timepoint-timepoint comparision
for i = 1:(number_of_timepoints - 1)
    figure()
    if mod(i,2) == 0

        G_t = cluster_segmented_photoreceptor{i,6}(2:end);
        S_t = cluster_segmented_photoreceptor{i,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t,S_t,'red','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on


        G_t_ = cluster_segmented_photoreceptor{i+1,6}(2:end);
        S_t_ = cluster_segmented_photoreceptor{i+1,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t_,S_t_,'blue','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on
        plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'r')
        plot(mean(G_t_,'all'),mean(S_t_,'all'),'o', 'MarkerFaceColor', 'b')


        axis equal
    else
        G_t = cluster_segmented_photoreceptor{i,6}(2:end);
        S_t = cluster_segmented_photoreceptor{i,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t,S_t,'blue','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on


        G_t_ = cluster_segmented_photoreceptor{i+1,6}(2:end);
        S_t_ = cluster_segmented_photoreceptor{i+1,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t_,S_t_,'red','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on
        plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'b')
        plot(mean(G_t_,'all'),mean(S_t_,'all'),'o', 'MarkerFaceColor', 'r')


        axis equal
    end
end

%% box plot without lifetime limit
% j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% % j = 3;
% % 3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% k = 1;%cluster
% hist_summary =[];
% hist_remap = [];
% hist_mean = [];
% for i = 1:number_of_timepoints
%     temp = cluster_results{i,j}{k}(cluster_results{i,j}{k}>0);
%     hist_summary= cat(1,hist_summary,temp);
%     hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
%     hist_mean = cat(1,hist_mean,mean(temp));
% end
% figure();
% clf
% boxplot(hist_summary,hist_remap)
% hold on
% plot(hist_mean, '*b')
% hold off


%% mean lifetime - for single cluster
for i = 1:number_of_timepoints
    figure()
    imagesc(cluster_segmented_photoreceptor{i,4})
    % Nonlinear cmap
    cMap = jet(256);
    dataMax = lifetime_limit(2);
    dataMin = lifetime_limit(1);
    centerPoint = 1.5;
    scalingIntensity = 1;
    %Then perform some operations to create your colormap. I have done this by altering the indices "x” at which each existing color lives, and then interpolating to expand or shrink certain areas of the spectrum.
    x = 1:length(cMap);
    x = x - (centerPoint-dataMin)*length(x)/(dataMax-dataMin);
    x = scalingIntensity * x/max(abs(x));
    %Next, select some function or operations to transform the original linear indices into nonlinear. In the last line, I then use "interp1” to create the new colormap from the original colormap and the transformed indices.
    x = sign(x).* exp(abs(x));
    x = x - min(x); x = x*511/max(x)+1;
    newMap = interp1(x, cMap, 1:512);
    %Then plot!
    colormap(newMap);
    clim(lifetime_limit);
    colorbar
    axis square
end

%% decay

% figure()
% plot()



% a = cluster_segmented_photoreceptor{1,5}
% plot(a{8})

% %% recalculate lifetime from decay
% 
% for i = 1:number_of_timepoints
%     num = cluster_segmented_photoreceptor{i,2};
%     if is_peak ==0
%         decay_matrix = zeros(num,98);
%     else
%         decay_matrix = zeros(num,71);
%     end
%     for j = 1:num
%         decay_matrix(j,:) = cluster_segmented_photoreceptor{i,5}{j};
%     end
%     [temp_g,temp_s] = PhasorTransform(decay_matrix,2);
% 
%     %out = scatplot(reshape(G_t,1,[]),reshape(S_t,1,[]));
%     %scatter_kde(reshape(G_t,1,[]),reshape(S_t,1,[]));
%     %scatter(reshape(G_t,1,[]),reshape(S_t,1,[]),'MarkerFaceColor','b','MarkerEdgeColor','b','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
% 
%     G_t = reshape(temp_g(2:end,:),1,[]);
%     S_t = reshape(temp_s(2:end,:),1,[]);
% 
% 
%     if is_adjust == 1
%         if is_peak ==0
%             % adjust phase and mod
%             %experiment location for 60x
%             S = 0.4251;
%             G = 0.1498;
%             % correct location
%             s = 0.4077;
%             g = 0.2110;
%         else
% 
%             %experiment location for 60x
%             S = 0.3439;
%             G = 0.1365;
%             % correct location
%             s = 0.3349;
%             g = 0.1289;
%         end
% 
%         phasediff=atan2(s,g)-atan2(S,G);
%         modfac=sqrt((s*s)+(g*g))/sqrt((S*S)+(G*G));
% 
% 
%         pha=atan2(S_t,G_t)+phasediff;
%         modu=sqrt((S_t.*S_t)+(G_t.*G_t))*modfac;
%         S_t=modu.*sin(pha); % calibration for ST- repeate for each channel
%         G_t=modu.*cos(pha); % calibration for GT- repeate for each channel
% 
%     end
% 
% 
%     figure()
%     omega = plot_PhasorCircle_with_reference_lifetime(1,71);
%     scatter(G_t,S_t);
%     scatplot(G_t',S_t')
% 
%     hold on
%     plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'r')
%     axis equal
% 
%     phasor_lifetime =(S_t./G_t)/omega;
%     figure()
%     histogram(phasor_lifetime(phasor_lifetime>lifetime_threshold))
%     figure()
%     boxplot(phasor_lifetime(phasor_lifetime>lifetime_threshold))
%     cluster_segmented_photoreceptor{i,8} = phasor_lifetime;
% end
% 
% 
% %% box plot for segmentation modify to visualize lifetime from decay
% lifetime_threshold = 0.5;
% hist_summary =[];
% hist_remap = [];
% hist_mean = [];
% for i = 1:number_of_timepoints
%     temp = cluster_segmented_photoreceptor{i,8}(cluster_segmented_photoreceptor{i,8}>lifetime_threshold &cluster_segmented_photoreceptor{i,8}<5)';
%     hist_summary= cat(1,hist_summary,temp);
%     hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
%     hist_mean = cat(1,hist_mean,mean(temp));
% end
% figure();
% clf
% boxplot(hist_summary,hist_remap)
% hold on
% plot(hist_mean, '*b')
% hold off
% 
% %% nhist modify to visualize lifetime from decay
% 
% nhist_summary = cell(number_of_timepoints,1);
% 
% for i = 1:number_of_timepoints
%     temp = cluster_segmented_photoreceptor{i,8}(cluster_segmented_photoreceptor{i,8}>lifetime_threshold &cluster_segmented_photoreceptor{i,8}<5)';
%     nhist_summary{i} = temp;
% end
% 
% % 'separate' to plot each set on its own axis, but with the same bounds
% % 'binfactor' change the number of bins used, larger value =more bins
% % 'samebins' force all bins to be the same for all plots
% % 'legend' add a legend in the graph (default for structs)
% % 'noerror' remove the mean and std plot from the graph
% % 'median' add the median of the data to the graph
% % 'text' return many details about each graph even if not plotted
% nhist(nhist_summary,'pdf','median','newfig','separate');
% nhist(nhist_summary,'pdf','median','newfig');
% nhist(nhist_summary,'median','newfig','separate','samebins','noerror','binfactor',2);
% nhist(nhist_summary,'median','newfig');
% 
% % %% combine intensity of 2 cluster RPE INTENSITY
% % combine_cluster = cluster_results{1,2}{2}+cluster_results{1,2}{3};
% % figure()
% % imagesc(combine_cluster)
% % axis square
% % %% combine lifetime of 2 cluster lIFETIME
% % 
% % combine_cluster = cluster_results{1,3}{2}+cluster_results{1,3}{3};
% % figure()
% % imagesc(combine_cluster)
% % axis square
% 
% % %% RPE segmentation
% % %[L_rpe, num_rpe] = RPE_segmentation_func(combine_cluster,10);
% % %% kNN
% % [L_rpe_kNN, num_rpe_kNN] = RPE_NC_segmentation(combine_cluster, L, num);
% % %% WNC
% % [L_rpe_WNC, num_rpe_WNC] = RPE_WNC_segmentation(combine_cluster, L, num, 1);

% %%
% b = 6
% combine_cluster = cluster_results{b,3}{1}+cluster_results{b,3}{3};
% figure()
% imagesc(combine_cluster)


%%
number_of_timepoints = 6;
RPE_segmentation_method = 2; % 1 - Watershed /2 - kNN /3 - WNC
m = [3 3 3 3 3 3 3 3];
%n = k(1:number_of_timepoints)
%old DBR FOV1
%n = [1 1 2 1 1 1];
%n = [2 2 2 2 2 2];

%n = [1 2 1 1 1 1];
n = m(1:number_of_timepoints) - k(1:number_of_timepoints)
for timepoint = 1:number_of_timepoints
    combine_cluster = cluster_results{timepoint,2}{m(timepoint)}+cluster_results{timepoint,2}{n(timepoint)};
    %combine_cluster = cluster_results{timepoint,2}{n(timepoint)};
    %combine_cluster = cluster_results{timepoint,2}{m(timepoint)};
    figure();
    imagesc(combine_cluster)
    % Photoreceptor segmentation
    L_photoreceptor =cluster_segmented_photoreceptor{timepoint,1};
    num_photoreptor = cluster_segmented_photoreceptor{timepoint,2};
    switch(RPE_segmentation_method)
        case 1
            [L, num] = RPE_segmentation_func(combine_cluster,10);
        case 2
            [L, num] = RPE_NC_segmentation(combine_cluster, L_photoreceptor, num_photoreptor);
        case 3
            [L, num] = RPE_WNC_segmentation(combine_cluster, L_photoreceptor, num_photoreptor,1);
    end
    cluster_segmented_RPE{timepoint,1} = L;
    cluster_segmented_RPE{timepoint,2} = num;
    % cluster numbered
    %0 is border
    %1 is back ground
    % figure();
    % imagesc(L);
    % axis square;

    % get average value for each cluster
    % figure();
    cluster_lifetime = cluster_results{timepoint,4}{m(timepoint)}+cluster_results{timepoint,4}{n(timepoint)};
    cluster_g = cluster_results{timepoint,5}(2:255,2:255);
    cluster_s = cluster_results{timepoint,6}(2:255,2:255);


    % imagesc(cluster_lifetime);
    % axis square;
    cluster_mean_lifetime = zeros(1,num);
    cluster_mean_g = zeros(1,num);
    cluster_mean_s = zeros(1,num);
    cluster_mean_lifetime_image = zeros(size(cluster_lifetime));
    cluster_decays = cell(1,num);

    for i = 2:num % start from 0 or 1
        %[row,col] = find(a==8)
        cluster_mask = L;
        cluster_mask(cluster_mask ~=i) = 0;
        cluster_mask(cluster_mask ==i) = 1;
        cluster_pixel_num = sum(cluster_mask(:) == 1);
        size(cluster_lifetime)
        size(cluster_mask)
        i
        timepoint
        cluster_lifetime_masked = cluster_lifetime.*cluster_mask;
        cluster_mean_lifetime(i) = sum(cluster_lifetime_masked,'all')/cluster_pixel_num;
        cluster_mean_lifetime_image = cluster_mean_lifetime_image + (cluster_mask)*cluster_mean_lifetime(i);

        cluster_g_masked = cluster_g.*cluster_mask;
        cluster_mean_g(i) = sum(cluster_g_masked,'all')/cluster_pixel_num;

        cluster_s_masked = cluster_s.*cluster_mask;
        cluster_mean_s(i) = sum(cluster_s_masked,'all')/cluster_pixel_num;
        %cluster_segmented_RPE{timepoint,3} = cluster_mean_lifetime;
        % figure();
        % imagesc(cluster_lifetime_masked)
        matrix_temp =top_matrix_gauss(2:255,2:255,:);
        single_element_matrix = bsxfun(@times, matrix_temp, cast(cluster_mask, 'like', matrix_temp));
        cluster_decays{i} = reshape(sum(single_element_matrix,[1 2]),1,[]);
        %cluster_decays{i} = calculate_decay(single_element_matrix);
    end
    %
    cluster_segmented_RPE{timepoint,3} = cluster_mean_lifetime;
    cluster_segmented_RPE{timepoint,4} = cluster_mean_lifetime_image;
    cluster_segmented_RPE{timepoint,5} = cluster_decays;
    cluster_segmented_RPE{timepoint,6} = cluster_mean_g;
    cluster_segmented_RPE{timepoint,7} = cluster_mean_s;

    lifetime_limit = [0 3];
    figure();
    histogram(cluster_mean_lifetime(2:end),20,'BinLimits',lifetime_limit);
    figure()
    boxplot(cluster_mean_lifetime(2:end))
end

%% box plot for segmentation

% set the above and below lifetime threshold to filter out outlier for
% better visualization
lifetime_threshold = 0.5;
hist_summary =[];
hist_remap = [];
hist_mean = [];
for i = 1:number_of_timepoints
    temp = cluster_segmented_RPE{i,3}(cluster_segmented_RPE{i,3}>lifetime_threshold & cluster_segmented_RPE{i,3}<1.5)';
    [hcounts{i,1}, hcounts{i,2}] = histcounts(temp,'BinWidth',binWidth); 
    hist_summary= cat(1,hist_summary,temp);
    hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
    hist_mean = cat(1,hist_mean,mean(temp));
end
maxCount = max([hcounts{:,1}]);
figure();
clf
boxplot(hist_summary,hist_remap)
hold on
plot(hist_mean, '*b')
hold off


%%
%%
load('excitation_scheme.mat');
figure();
clf
%pos = [1, 10, 30, 40, 60, 70];
pos = [2.5, 12.5, 32.5, 42.5, 62.5, 72.5];
yyaxis left
colors = lines(length(pos)); 
boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
hold on
scatter(pos,hist_mean,50,colors,"filled")
plot(pos,hist_mean, 'LineWidth', 1,'Color','r','LineStyle','- -');

xInterval = 1; %x-interval is always 1 with boxplot groups
%normwidth = (1-hgapGrp-hgap)/2;    
binWidth = 0.01;  % histogram bin widths
hgapGrp = .15;   % horizontal gap between pairs of boxplot/histograms (normalized)
hgap = 0.06;      % horizontal gap between boxplot and hist (normalized)
xCoordinate = pos;  %x-positions is always 1:n with boxplot groups
histX0 = xCoordinate + 5/2;    % histogram base
maxHeight = 4;       % max histogram height
patchHandles = gobjects(1,length(pos)); 
for i = 1:length(pos)
    % Normalize heights 
    height = hcounts{i,1}/maxCount*maxHeight;
    % Compute x and y coordinates 
    xm = [zeros(1,numel(height)); repelem(height,2,1); zeros(2,numel(height))] + histX0(i);
    yidx = [0 0 1 1 0]' + (1:numel(height));
    ym = hcounts{i,2}(yidx);
    % Plot patches
    patchHandles(i) = patch(xm(:),ym(:),colors(i,:),'EdgeColor',colors(i,:),'LineWidth',1,'FaceAlpha',.45);
end
xlim([-5 85])
%ylim([0.5 0.7])
hold on

%figure();
%clf
%pos = [1, 10, 30, 40, 60, 70];
%colors = lines(length(pos)); 
% % yyaxis left
% % boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
% % hold on
% % scatter(pos,hist_mean,50,colors,"filled")
% % plot(pos,hist_mean, 'LineWidth', 1,'Color','r','LineStyle','--');
yyaxis right
plot(excitation_scheme(:,1), excitation_scheme(:,2),'Color','r','LineWidth',2)
ylim([0 10])
xticks(0:5:75);
xticklabels(0:5:75);
hold off



%%
figure();
clf
pos = [1, 10, 30, 40, 60, 70];
colors = lines(length(pos)); 
yyaxis left
boxplot(hist_summary,hist_remap, 'positions', pos, 'widths', 5, 'colors', colors)
hold on
scatter(pos,hist_mean,50,colors,"filled")
plot(pos,hist_mean, 'LineWidth', 1,'Color','r');
yyaxis right
plot(excitation_scheme(:,1), excitation_scheme(:,2))
ylim([0 1.2])
hold off

%% nhist

% set the above and below lifetime threshold to filter out outlier for
% better visualization

nhist_summary = cell(number_of_timepoints,1);

for i = 1:number_of_timepoints
    temp = cluster_segmented_RPE{i,3}(cluster_segmented_RPE{i,3}>lifetime_threshold & cluster_segmented_RPE{i,3}<1)';
    nhist_summary{i} = temp;
end

% 'separate' to plot each set on its own axis, but with the same bounds
% 'binfactor' change the number of bins used, larger value =more bins
% 'samebins' force all bins to be the same for all plots
% 'legend' add a legend in the graph (default for structs)
% 'noerror' remove the mean and std plot from the graph
% 'median' add the median of the data to the graph
% 'text' return many details about each graph even if not plotted
nhist(nhist_summary,'pdf','median','newfig','separate');
nhist(nhist_summary,'pdf','median','newfig');
nhist(nhist_summary,'median','newfig','separate','samebins','noerror','binfactor',2);
nhist(nhist_summary,'median','newfig');

%%
for timepoint = 1:number_of_timepoints
    G_t = cluster_segmented_RPE{timepoint,6}(2:end);
    S_t = cluster_segmented_RPE{timepoint,7}(2:end);
    figure()
    omega = plot_PhasorCircle_with_reference_lifetime(1,71);
    scatter(G_t,S_t);

    %scatplot(G_t',S_t')
    hold on
    plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'r')
    axis equal
    lifetime_mean = (mean(S_t,'all')/mean(G_t,'all'))/omega;
    formatSpec = 'timepoint %i: G_t is %4.4f - S_t is %4.4f mm\n- mean lifetime is %4.4f ns\n';
    fprintf(formatSpec,timepoint,mean(G_t,'all'),mean(S_t,'all'),lifetime_mean)
end
%% timepoint-timepoint comparision
for i = 1:(number_of_timepoints - 1)
    figure()
    if mod(i,2) == 0

        G_t = cluster_segmented_RPE{i,6}(2:end);
        S_t = cluster_segmented_RPE{i,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t,S_t,'red','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on


        G_t_ = cluster_segmented_RPE{i+1,6}(2:end);
        S_t_ = cluster_segmented_RPE{i+1,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t_,S_t_,'blue','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on
        plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'r')
        plot(mean(G_t_,'all'),mean(S_t_,'all'),'o', 'MarkerFaceColor', 'b')


        axis equal
    else
        G_t = cluster_segmented_RPE{i,6}(2:end);
        S_t = cluster_segmented_RPE{i,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t,S_t,'blue','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on


        G_t_ = cluster_segmented_RPE{i+1,6}(2:end);
        S_t_ = cluster_segmented_RPE{i+1,7}(2:end);

        omega = plot_PhasorCircle_with_reference_lifetime(1,71);
        scatter(G_t_,S_t_,'red','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2);
        %scatplot(G_t',S_t')
        hold on
        plot(mean(G_t,'all'),mean(S_t,'all'),'o', 'MarkerFaceColor', 'b')
        plot(mean(G_t_,'all'),mean(S_t_,'all'),'o', 'MarkerFaceColor', 'r')


        axis equal
    end
end

%% box plot select clusters
% %% box plot
% j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% %j = 3;
% %3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% k = [1 1 1 1 1 1 1 1];%cluster
% hist_summary =[];
% hist_remap = [];
% hist_mean = [];
%
% % rng default  % For reproducibility
% % x = [randn(25,4);rand(2,4)-6;rand(2,4)+6];
% % x = reshape(x(randperm(numel(x))),size(x)); % scrambles rows of x; for demo purposes only
% % isout = isoutlier(x,'quartiles');
% % xClean = x;
% % xClean(isout) = NaN;
%
% for i = 1:number_of_timepoints
%     temp = cluster_results{i,j}{k(i)}(cluster_results{i,j}{k(i)}>0);
%     hist_summary= cat(1,hist_summary,temp);
%     hist_remap = cat(1,hist_remap,repmat({string(i)},size(temp)));
%     hist_mean = cat(1,hist_mean,mean(temp));
% end
% figure();
% clf
% boxplot(hist_summary,hist_remap)
% hold on
% plot(hist_mean, 'dg')
% hold off




% % % nhist
% %
% % j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% % j = 3
% % 3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% % k = 1;%cluster
% % nhist_summary = cell(number_of_timepoints,1);
% %
% % for i = 1:number_of_timepoints
% %     temp = cluster_results{i,j}{k}(cluster_results{i,j}{k}>0);
% %     nhist_summary{i} = temp;
% % end
% % nhist(nhist_summary,'pdf','median','newfig','separate');
% % nhist(nhist_summary,'pdf','median','newfig');
% % nhist(nhist_summary,'median','newfig','separate');
% % nhist(nhist_summary,'median','newfig');



% %% nhist select
%
% j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% % j = 3
% % 3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% k = [1 1 1 1 1 1 1 1];%cluster
% nhist_summary = cell(number_of_timepoints);
%
% for i = 1:number_of_timepoints
%     temp = cluster_results{i,j}{k(i)}(cluster_results{i,j}{k(i)}>0);
%     if outliers_remove == 1
%         isout = isoutlier(temp,'quartiles');
%         temp(isout) = NaN;
%         isout = isoutlier(temp,'quartiles');
%         temp(isout) = NaN;
%         isout = isoutlier(temp,'quartiles');
%         temp(isout) = NaN;
%         isout = isoutlier(temp,'quartiles');
%         temp(isout) = NaN;
%         isout = isoutlier(temp,'quartiles');
%         temp(isout) = NaN;
%     end
%     nhist_summary{i} = temp;
% end
% nhist(nhist_summary,'pdf','median','newfig','separate');
% nhist(nhist_summary,'pdf','median','newfig');
%
% %% nhist - 2 time point
% j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% % j = 3
% % 3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% k = [1 1 1 1 1 1 1 1];%cluster
% nhist_summary = cell(2);
% for i = 1:(number_of_timepoints-1)
%     if mod(i,2) == 0
%         temp = cluster_results{i,j}{k(i)}(cluster_results{i,j}{k(i)}>0);
%         nhist_summary{1} = temp;
%         temp = cluster_results{i+1,j}{k(i+1)}(cluster_results{i+1,j}{k(i+1)}>0);
%         nhist_summary{2} = temp;
%     else
%         temp = cluster_results{i,j}{k(i)}(cluster_results{i,j}{k(i)}>0);
%         nhist_summary{2} = temp;
%         temp = cluster_results{i+1,j}{k(i+1)}(cluster_results{i+1,j}{k(i+1)}>0);
%         nhist_summary{1} = temp;
%     end
%     nhist(nhist_summary,'pdf','median','newfig','separate');
%     nhist(nhist_summary,'median','newfig');
% end
%
% %
%
% for i = 1:3
%     cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}==0) = NaN;
% end
% nhist(cluster_filtered_phasor_lifetime,'pdf','median','newfig','separate');
% %
% figure();
% clf
% boxplot(cluster_results{i,j}{k}(cluster_results{i,j}{k}>0))
% hold on
%
% i = 2;%time point
% j = 4;% variable: 1-cluster proble; 2-cluster intensity;
% % 3-cluster_filtered_phasor_lifetime; 4-cluster_filtered_phasor_lifetime_without_limits
% k = 1;%cluster
% boxplot(cluster_results{i,j}{k}(cluster_results{i,j}{k}>0))
% hold on
%
% %
% figure()
% clf
% boxplot(cluster_filtered_phasor_lifetime{1})
%
% %% histogram visualization
% i=1;
% hist_fig = figure();
% cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}<lifetime_limit(1)) = 0;
% cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>lifetime_limit(2)) = 0;
% average_lifetime = mean(cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>0),'all');
%
% fprintf('Group %d average lifetime: %d ns\n',i,average_lifetime);
%
% cluster_histogram = histogram(cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>0),60,'BinLimits',lifetime_limit);
% [~, idx] = max(cluster_histogram.Values); % tallest bin(y value) and its index location
% mostLikely = cluster_histogram.BinEdges(idx); % x value associated with greatest y value
% fprintf('Group %d peak lifetime: %d ns\n',i,mostLikely);
%
%
% hist_fig_savefile = strcat('lifetime_histogram_cluster ',string(i),'.png');
% saveas(hist_fig, hist_fig_savefile);
% %%
% clf
% boxplot(cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>0))
% %%
% nhist(cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>0))
% %%
% nhist(cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}>0),'color','jet')
%
% %%
% %close all
% for i = 1:3
%     cluster_filtered_phasor_lifetime{i}(cluster_filtered_phasor_lifetime{i}==0) = NaN;
% end
% nhist(cluster_filtered_phasor_lifetime,'pdf','median','newfig');
%
% %%
%
% close all
% % Cell array example:
% A={randn(1,10^5),randn(10^3,1)+1};
% nhist(A,'legend',{'\mu=0','\mu=1'},'separate');
% nhist(A,'legend',{'\mu=0','\mu=1'});
%
% %%
% A=[randn(1,10^5)+1 randn(1,2*10^5)+5];
% nhist(A,'pdf','color','colormap')
%
% %% Structure example:
% A.mu_is_Zero=randn(1,10^5); A.mu_is_Two=randn(10^3,1)+2;
% nhist(A);
% nhist(A,'color','summer')
% nhist(A,'color',[.3 .8 .3],'separate')
% nhist(A,'binfactor',4)
% nhist(A,'samebins')
% nhist(A,'median','noerror')