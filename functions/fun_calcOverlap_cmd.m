function handles = fun_calcOverlap_cmd(handles,K,overlay_intensity)
%FUN_CALCOVERLAP Summary of this function goes here
%   This function is used to calculate the overlap images
%
%   Author: Yide Zhang
%   Email: yzhang34@nd.edu
%   Date: April 12, 2019
%   Copyright: University of Notre Dame, 2019

if isfield(handles, 'Clusteridx') && isfield(handles, 'xyzgood')
    
    I_stack = handles.imageI;
    xyz_good = handles.xyzgood; 
    Cluster_idx = handles.Clusteridx; 
    [~, ~, n_z] = size(I_stack);
    
    cc = fun_HSVcolors(K, 1);
    
    
        Imin = min(I_stack(:));
        Imax = max(I_stack(:));
    
    
    % convert to gray scale
    I_stack = mat2gray(I_stack,[Imin Imax]);
    O_stack = zeros(size(I_stack,1), size(I_stack,2), 3, n_z);    
    for i_z = 1:n_z  
        O_stack(:,:,:,i_z) = repmat(I_stack(:,:,i_z),[1 1 3]);
    end
    OHSV_stack = O_stack;
        
    hue_max = 0.7;
    hue_min = 0;
    K_hue = linspace(hue_max, hue_min, K)'; % scale of hue

    hwb_progress = waitbar(0, 'Calculating overlap ...');
    for iK = 1:K
        waitbar(iK/K, hwb_progress);
        if size(xyz_good, 1) >= K
            x_iK = xyz_good(Cluster_idx==iK,1);
            y_iK = xyz_good(Cluster_idx==iK,2);
            z_iK = xyz_good(Cluster_idx==iK,3);
        else
            x_iK = 1;
            y_iK = 1;
            z_iK = 1;
        end
        n_good = numel(x_iK);
        I_good = zeros(n_good, 1);
        for i_good = 1:n_good
            I_good(i_good) = I_stack(x_iK(i_good), y_iK(i_good), z_iK(i_good));
        end
        map_hue = K_hue(iK) * ones(n_good,1);
        map_saturation = ones(n_good,1);
        map_value = I_good;
        cc_I = hsv2rgb([map_hue map_saturation map_value]);
        for i_good = 1:n_good
            O_stack(x_iK(i_good), y_iK(i_good), :, z_iK(i_good)) = cc(iK, :);
            OHSV_stack(x_iK(i_good), y_iK(i_good), :, z_iK(i_good)) = cc_I(i_good, :);
        end
    end
    close(hwb_progress);
    
    handles.imageO = O_stack; 
    handles.imageOHSV = OHSV_stack; 
%    guidata(hObject,handles) 

    
else
    msgbox('Please calculate clusters first.', 'Error','error');
end

if isfield(handles, 'imageI')
            I_stack = handles.imageI;
            [~,~,slice_n] = size(I_stack);  

            if slice_n == 1
            
            figure();
            if isfield(handles, 'imageO') && isfield(handles, 'imageOHSV') 
                O_stack = handles.imageO;
                OHSV_stack = handles.imageOHSV; 
                if overlay_intensity
                    imshow(OHSV_stack(:,:,:));  
                else
                imshow(O_stack(:,:,:));  
                end
            else
                msgbox('Please calculate clusters first.', 'Error','error');
            end
        end

end

