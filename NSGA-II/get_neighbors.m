function [all_neighbors, occupied_neighbors, empty_neighbors] = get_neighbors(voxel_number,design, direction_to_apply_periodicity)
    sz = size(design);
    [i,j,k] = ind2sub(sz,voxel_number);
    voxel_indices = [i j k];

    % voxel_index is a row vector of size [1 3] containing the row,
    % column, and page of the voxel whose neighbors we are seeking.
    directions_to_check = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1]; % Look in positive and negative directions for each axis
    neighbor_inds_to_check = voxel_indices + directions_to_check;
    
    % Take care of neighbors that have out of range indices (zero or sz(i) + 1)
    for i = 1:3
        % temp = neighbor_inds_to_check(:,i);
        % temp(temp == 0) = sz(i);
        % neighbor_inds_to_check(:,i) = temp;
        if direction_to_apply_periodicity(i)
            neighbor_inds_to_check(neighbor_inds_to_check(:,i) == 0,i) = sz(i); % Wrap around to the end
            neighbor_inds_to_check(neighbor_inds_to_check(:,i) == sz(i)+1,i) = 1; % Wrap around to the beginning
        else
            neighbor_inds_to_check(neighbor_inds_to_check(:,i) == 0,i) = []; % Eliminate out-of-range neighbor inds
            neighbor_inds_to_check(neighbor_inds_to_check(:,i) == sz(i)+1,i) = []; % Eliminate out-of-range neighbor inds
        end
    end
    
    % Find linear indices of neighbors (with and without materials)
    all_neighbors = sub2ind(sz,neighbor_inds_to_check(:,1),neighbor_inds_to_check(:,2),neighbor_inds_to_check(:,3));

    % Extract the linear indices of the voxels that contain material
    occupied_neighbors = all_neighbors(logical(design(all_neighbors)));
    
    % The voxels without material are the neighbors
    empty_neighbors = setdiff(all_neighbors, occupied_neighbors);
end