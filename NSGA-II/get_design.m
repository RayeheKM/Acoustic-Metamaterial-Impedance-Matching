function design = get_design(design_params)
    if design_params.design_options{1}.isOnlyGenerateOnePropertyField
        prop_idx = 1;
        design = get_prop(design_params,prop_idx);
    else
        design = zeros([design_params.N_pix 3]);
        for prop_idx = 1:3
            design(:,:,:,prop_idx) = get_prop(design_params,prop_idx);
        end
    end
end
