function [pfc, CenterNodes, BoundaryOK, Voxels, group_data] = generate_connected_design(rng_seed_offset, GridNumber,Size)
    % clear; close all;
    
    mfn = mfilename;
    
    N_struct = 1;
%     rng_seed_offset = 0;
    cmap = 'parula';
    isShowTicks = false;
    isShowAxes = false;
    isUsePresetFigurePosition = false;
    isExportPdf = false;
    isExportPng = false;
    PngResolution = 1000;
    isSetBackgroundColor = false;
    % BackgroundColor = uint8([218,227,243]); % mylightblue
    % ConnectedGroupColor = uint8([24 55 77]); % mydarkblue
    % UnconnectedGroupColor = uint8([50 100 240]); % mymediumblue
    isSaveDesigns = false;
    
    isUseGPU = false;
    
    length_scale = [1 1 1];
    
    const.a = 1; % [m]
    const.N_pix = [GridNumber GridNumber GridNumber];
    
    design_params = design_parameters;
    design_params.design_number = []; % leave empty
    design_params.design_style = 'kernel';
    design_params.design_options = struct('kernel','periodic','period',[1 1 1],'sigma_f',1,'sigma_l',length_scale,'symmetry_type','none','N_value',2,'isUseGPU',isUseGPU,'isOnlyGenerateOnePropertyField',true);
    design_params.N_pix = const.N_pix;
    design_params = design_params.prepare();
    
    const.E_min = 200e6; % 2e9
    const.E_max = 200e9; % 200e9
    const.rho_min = 8e2; % 1e3
    const.rho_max = 8e3; % 8e3
    const.poisson_min = 0; % 0
    const.poisson_max = .5; % .5
    const.t = 1;
    const.sigma_eig = 1;
    
    const.symmetry_type = 'c1m1'; IBZ_shape = 'rectangle'; num_tesselations = 1;
    
    const.design_scale = 'linear';
    const.design = nan([const.N_pix 3]); % This is just a temporary value so that 'const' has the field 'design' used in the parfor loop
    
    %% Generate designs
    % if isUsePresetFigurePosition
    %     data = load([mfn '_preset_figure_position']);
    %     preset_figure_position = data.preset_figure_position;
    %     fig = figure('Position',preset_figure_position);
    % else
    %     fig = figure;
    % end
    
    % tlo = tiledlayout('flow');
    % tlo.TileSpacing = 'compact';
    for struct_idx = 1:N_struct
        pfc = const;
        pfdp = design_params;
    
        pfdp.design_number = struct_idx + rng_seed_offset;
        pfdp = pfdp.prepare();
        pfc.design = get_design(pfdp);
        pfc.design = convert_design(pfc.design,'linear',pfc.design_scale,pfc.E_min,pfc.E_max,pfc.rho_min,pfc.rho_max);

        pfc.design(1,:,:) = pfc.design(end,:,:);
        pfc.design(:,1,:) = pfc.design(:,end,:);
        pfc.design(:,:,1) = pfc.design(:,:,end);
        
          
        [group_data, isConnected, Voxel]  = find_connected_group(gather(pfc.design));
        % group_datas{struct_idx} = group_data;
        % evolution_histories{struct_idx} = evolution_history;
    
        % ax = nexttile;
        % property_idx = 1;
        % options = struct();
        % options.layer_spacing = 0;
        % options.color = DesignColor;
        % visualize_single_material_design(pfc.design,property_idx,options,ax)
        % options.group_colors = {ConnectedGroupColor,UnconnectedGroupColor};
        % groups = {group_data.connected_group,group_data.unconnected_group};
        
        % visualize_single_material_design_with_groups(pfc.design,groups,options,ax);
    
        % colormap(cmap)
        % if ~isShowTicks
        %     set(ax,'XTickLabel',[])
        %     set(ax,'YTickLabel',[])
        %     set(ax,'ZTickLabel',[])
        %     set(ax,'XTick',[])
        %     set(ax,'YTick',[])
        %     set(ax,'ZTick',[])
        % end
        % if ~isShowAxes
        %     set(ax,'Visible','off')
        % end
    
        %     colorbar
    end
    
    % if isSetBackgroundColor
    %     set(fig,'color',BackgroundColor)
    % end
    % 
    % % Export figure
    % if isExportPdf
    %     exportgraphics(fig,[mfn '_export.pdf'],'contenttype','vector','BackgroundColor','current')
    % end
    % 
    % if isExportPng
    %     exportgraphics(fig,[mfn '_export.png'],'resolution',PngResolution,'BackgroundColor','current')
    % end
    
    % Save preset_figure_position
    % preset_figure_position = fig.Position;
    % save([mfn '_preset_figure_position'],"preset_figure_position")

    if length(group_data.connected_group)>length(group_data.unconnected_group)
        longest_connected_group=group_data.connected_group;
    else
        pfc.design1=pfc.design;
        group_data1=group_data;
        check_group_data = {};
        check_group_data {end+1}={group_data1.connected_group};
        while ~isempty(group_data1.unconnected_group)
            row=0;
            column=0;
            page=0;
            [row, column, page] = ind2sub(size(pfc.design1), group_data1.connected_group);
            pfc.design1(row, column, page)=0;
            [group_data1, isConnected, Voxel1]  = find_connected_group(gather(pfc.design1));
            check_group_data{end+1} = group_data1.connected_group;
        end
        
        group_lengths = cellfun(@length, check_group_data);
        [~, longest_index] = max(group_lengths);
        longest_connected_group = check_group_data{longest_index};
    end
    
    indices_set_to_zero=setdiff(1:numel(pfc.design), longest_connected_group);
    pfc.design(indices_set_to_zero)=0;
    [group_data, isConnected, Voxels]  = find_connected_group(gather(pfc.design));

    % disp(group_data)

    designs(struct_idx,:,:,:) = pfc.design;

    designs = gather(designs);

    GeometryData = designs;
    [a,b,c] = ind2sub(size(pfc.design), find(pfc.design==1));
    

    Number_of_Cubes = length(a);
    CenterNodes = zeros(Number_of_Cubes,3);

    for j=1:length(a)
        CenterNodes(j,:) = Size/GridNumber.*[a(j)-1+0.5, b(j)-1+0.5, c(j)-1+0.5];
    end
    

    % check if we have atleat one cube on the boundaries of each direction
    
    cubeOnX = any(CenterNodes(:, 1)==0.5*Size/GridNumber);
    cubeOnY = any(CenterNodes(:, 2)==0.5*Size/GridNumber);
    cubeOnZ = any(CenterNodes(:, 3)==0.5*Size/GridNumber);
    
    if cubeOnX && cubeOnY && cubeOnZ
        % disp('At least one cube exists on the x=0, y=0, and z=0 planes.');
        BoundaryOK = true;
    else
        % disp('At least one cube does not exist on one of the planes (x=0, y=0, or z=0).');
        BoundaryOK = false;
    end

end