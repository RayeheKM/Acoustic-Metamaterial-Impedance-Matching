function [group_data, isConnected, Voxel] = find_connected_group(design)
    sz = size(design,[1 2 3]);
    num_voxels = prod(sz);
    voxel_numbers = reshape(1:num_voxels,sz);

    material_voxel_numbers = find(design);
    num_material_voxels = nnz(design);

    void_voxel_numbers = find(~design);
    num_void_voxels = numel(void_voxel_numbers);

    % Pick a starting voxel from the material_voxels
    % starting_material_voxel_number = randi(num_material_voxels);
    starting_material_voxel_number = 1;
    starting_voxel_number = material_voxel_numbers(starting_material_voxel_number);

    % Initialize evolution_data
    if nargout == 2
        evolution_data = struct();
        evolution_data.connected_group_new_history = {};
        evolution_data.neighbor_group_history = {};
        evolution_data.connected_group_history = {};
    end
    
    all_empty_neighbors = [];

    % Loop through iterations to find all connected neighbors
    connected_group = starting_voxel_number;
    connected_group_new = starting_voxel_number; % Only the members of connected_group that were newly added in the last iteration
    new_members_are_found = true; % Initialize
    index_voxels = 1;
    while new_members_are_found
        neighbor_group = [];
        for i = 1:length(connected_group_new)
            voxel_number = connected_group_new(i);
            [all_neighbors, occupied_neighbors, empty_neighbors] = get_neighbors(voxel_number,design);
            % Save neighbor information in Voxel struct
            Voxel(index_voxels).voxel_number = voxel_number;
            Voxel(index_voxels).all_neighbors = all_neighbors;
            Voxel(index_voxels).occupied_neighbors = occupied_neighbors;
            Voxel(index_voxels).empty_neighbors = empty_neighbors;
            index_voxels = index_voxels+1;
            neighbor_group = [neighbor_group occupied_neighbors'];
            all_empty_neighbors = [all_empty_neighbors empty_neighbors'];
        end
        connected_group_new = setdiff(neighbor_group,connected_group);
        connected_group = [connected_group connected_group_new];

        if nargout == 2
            evolution_data.connected_group_new_history{end+1} = connected_group_new;
            evolution_data.neighbor_group_history{end+1} = connected_group_new;
            evolution_data.connected_group_history{end+1} = connected_group;
        end

        if isempty(connected_group_new)
            new_members_are_found = false;
        end
    end


    group_data.connected_group = connected_group;
    group_data.unconnected_group = reshape(setdiff(material_voxel_numbers,connected_group),1,[]);
    group_data.all_empty_neighbors = unique(all_empty_neighbors);

    isConnected = isempty(group_data.unconnected_group);

end