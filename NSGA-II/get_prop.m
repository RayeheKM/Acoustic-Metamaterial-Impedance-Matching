function prop = get_prop(design_parameters,prop_idx)
    design_style = design_parameters.design_style{prop_idx};
    design_options = design_parameters.design_options{prop_idx};
    N_pix = design_parameters.N_pix;
    if ~isempty(design_parameters.design_number)
        design_number = design_parameters.design_number(prop_idx);
    end

    switch design_style
        case 'constant'
            prop = design_options.constant_value*ones(N_pix);
        case 'uncorrelated'
            rng(design_number,'twister')
            prop = rand(N_pix);
        case 'kernel'
            gpurng(design_number) % 'twister' doesn't exist on GPU as of now
            rng(design_number,'twister')
            prop = kernel_prop(design_options.kernel,N_pix,design_options);
        case 'diagonal-band'
            error('not implemented')
            prop = eye(design_parameters.N_pix);
            for i = 1:design_options.feature_size
                prop = prop + diag(ones(N_pix(1)-i,1),i);
                prop = prop + diag(ones(N_pix(1)-i,1),-i);
            end
        case 'dispersive-tetragonal'
            error('not implemented')
            % Dispersive cell - Tetragonal
            N_pix_inclusion = design_options.feature_size;
            prop = zeros(N_pix);
            mask = zeros(N_pix(1),1);
            mask(1:N_pix_inclusion) = 1;
            mask = circshift(mask,round((N_pix(1) - N_pix_inclusion)/2));
            idxs = find(mask);
            %             idxs = round(N_pix/4 + 1):round(3*N_pix/4);
            prop(idxs,idxs) = 1;
        case 'dispersive-tetragonal-negative'
            error('not implemented')
            % Dispersive cell - Tetragonal
            prop = zeros(N_pix); % the first pane is E
            mask = round(N_pix/4 + 1):round(3*N_pix/4);
            prop(mask,mask) = 1;
            prop = ~prop; % negative!
        case 'dispersive-orthotropic'
            error('not implemented')
            % Dispersive cell - Orthotropic
            prop = zeros(N_pix); % the first pane is E
            mask = (N_pix/4 + 1):(3*N_pix/4);
            prop(:,mask) = 1;
        case 'homogeneous'
            error('not implemented')
            % Homogeneous cell
            prop = ones(N_pix); % the first pane is E
        case 'quasi-1D'
            error('not implemented')
            % Quasi-1D cell
            prop = ones(N_pix);
            prop(:,1:2:end) = 0;
        case 'rotationally-symmetric'
            error('not implemented')
            prop = zeros(N_pix);
            mask = (N_pix/4 + 1):(2*N_pix/4);
            prop(mask,mask) = 1;
            mask = (2*N_pix/4 + 1):(3*N_pix/4);
            prop(mask,mask) = 1;
        case 'sierpinski'
            error('not implemented')
            ratio = N_pix(1)/30;
            prop = ones(N_pix);

            mask = false(size(prop));
            cols = (ratio*2 + 1):(ratio*14);
            rows = (ratio*(30 - 14) + 1):(ratio*(30-2));
            mask(rows,cols) = triu(true(ratio*12));
            prop(mask) = 0;

            mask = false(size(prop));
            cols = (ratio*4 + 1):(ratio*28);
            rows = (ratio*(30 - 28) + 1):(ratio*(30-4));
            mask(rows,cols) = triu(true(ratio*24));
            prop(mask) = 0;
        otherwise
            error(['design not recognized: ' design_style])
    end
    if isfield(design_options,'symmetry')
        switch design_options.symmetry
            case 'c1m1'
                error('not implemented')
                orig_min = min(prop,[],'all');
                orig_range = range(prop,'all');
                prop = 1/2*prop + 1/2*prop';
                new_range = range(prop,'all');
                prop = orig_range/new_range*prop;
                new_min = min(prop,[],'all');
                prop = prop - new_min + orig_min;
            case 'none'
                % do nothing
            otherwise
                error('symmetry_type not recognized')
        end
    end
    if isfield(design_options,'N_value') && design_options.N_value ~= inf
        prop = round((design_options.N_value - 1)*prop)/(design_options.N_value - 1);
    end
end