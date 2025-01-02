function [popc2, RFlag] = Remove(p2, GridNumber, Size)
    
    CenterNodes = p2.CenterNodes;
    group_data = p2.group_data;
    Voxels = p2.Voxels;
    pfc = p2.pfc;
    
    disp(length(Voxels))

    connectivity_check = true;
    checkcodeNumber=1;
    Removed_CenterNodes = [];
    NotAdmissableCubes = [];
    
    while connectivity_check
        design_check = pfc.design;
        voxels_check = [Voxels.voxel_number];
        voxels_check = setdiff(voxels_check, NotAdmissableCubes);
        if isempty(voxels_check)
            disp('No voxels')
            % popc2 = Add(p2, GridNumber, Size);
            popc2 = p2;
            RFlag = false;
            return;
        end
        remove_index_idx = randi(length(voxels_check));
        remove_index = voxels_check(remove_index_idx);
        [row, column, page] = ind2sub(size(design_check), remove_index);
    
        if row == 1 || row == size(design_check,1)
            if column == 1 || column == size(design_check,1)
                if page == 1 || page == size(design_check,1)
                    % disp('Corner - 8 voxels');
                    rows = [row, size(design_check,1)+1-row, row, row, size(design_check,1)+1-row, size(design_check,1)+1-row, row, size(design_check,1)+1-row];
                    columns = [column, column, size(design_check,1)+1-column, column, size(design_check,1)+1-column, column, size(design_check,1)+1-column, size(design_check,1)+1-column];
                    pages = [page, page, page, size(design_check,1)+1-page, page, size(design_check,1)+1-page, size(design_check,1)+1-page, size(design_check,1)+1-page];

                    design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

                    % design_check(row, column, page) = 0;
                    % design_check(size(design_check,1)+1-row, column, page) = 0;
                    % design_check(row, size(design_check,1)+1-column, page) = 0;
                    % design_check(row, column, size(design_check,1)+1-page) = 0;
                    % design_check(size(design_check,1)+1-row, size(design_check,1)+1-column, page) = 0;
                    % design_check(size(design_check,1)+1-row, column, size(design_check,1)+1-page) = 0;
                    % design_check(row, size(design_check,1)+1-column, size(design_check,1)+1-page) = 0;
                    % design_check(size(design_check,1)+1-row, size(design_check,1)+1-column, size(design_check,1)+1-page) = 0;
    
                    [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
                    if length(group_data_new.connected_group) == (length(group_data.connected_group)-8)
                        % disp('If')
                        connectivity_check = false;

                        pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 0;

                        % pfc.design(row, column, page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, column, page) = 0;
                        % pfc.design(row, size(pfc.design,1)+1-column, page) = 0;
                        % pfc.design(row, column, size(pfc.design,1)+1-page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, column, size(pfc.design,1)+1-page) = 0;
                        % pfc.design(row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 0;
    
                        nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5];
                            Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, size(design_check,1)+1-page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, size(design_check,1)+1-column-1+0.5, size(design_check,1)+1-page-1+0.5] ];
    
                        Voxels = Voxels_new;
                        group_data = group_data_new;
    
                        Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                        CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-7:end,:),'rows');
    
                    else
                        % disp('Else')
    
                        remove_index_idxs = [
                            row, column, page;
                            size(design_check,1) + 1 - row, column, page;
                            row, size(design_check,2) + 1 - column, page;
                            row, column, size(design_check,3) + 1 - page;
                            size(design_check,1) + 1 - row, size(design_check,2) + 1 - column, page;
                            size(design_check,1) + 1 - row, column, size(design_check,3) + 1 - page;
                            row, size(design_check,2) + 1 - column, size(design_check,3) + 1 - page;
                            size(design_check,1) + 1 - row, size(design_check,2) + 1 - column, size(design_check,3) + 1 - page;
                            ];
    
                        % Calculate the reverse indices
                        remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                        % remove_voxelss = voxels_check(remove_index_numbers);
    
                        NotAdmissableCubes (end+1:end+8) =  remove_index_numbers;
                        NotAdmissableCubes = unique(NotAdmissableCubes);
                        % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                        % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                        checkcodeNumber=checkcodeNumber+1;
    
                    end
    
    
                else
                    % disp('XY edge - 4 voxels');

                    rows = [row, size(design_check,1)+1-row, row, size(design_check,1)+1-row];
                    columns = [column, column, size(design_check,1)+1-column, size(design_check,1)+1-column];
                    pages = [page, page, page, page];

                    design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;


                    % design_check(row, column, page) = 0;
                    % design_check(size(design_check,1)+1-row, column, page) = 0;
                    % design_check(row, size(design_check,1)+1-column, page) = 0;
                    % design_check(size(design_check,1)+1-row, size(design_check,1)+1-column, page) = 0;
    
                    [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
                    if length(group_data_new.connected_group) == (length(group_data.connected_group)-4)
                        % disp('If')
                        connectivity_check = false;

                        pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                        % pfc.design(row, column, page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, column, page) = 0;
                        % pfc.design(row, size(pfc.design,1)+1-column, page) = 0;
                        % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, page) = 0;
    
                        nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5];
                            Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5]];
    
                        Voxels = Voxels_new;
                        group_data = group_data_new;
    
                        Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                        CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-3:end,:),'rows');
    
                    else
                        % disp('Else')
    
                        remove_index_idxs = [
                            row, column, page;
                            size(design_check,1) + 1 - row, column, page;
                            row, size(design_check,2) + 1 - column, page;
    
                            size(design_check,1) + 1 - row, size(design_check,2) + 1 - column, page;
    
                            ];
    
                        % Calculate the reverse indices
                        remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                        % remove_voxelss = voxels_check(remove_index_numbers);
    
                        NotAdmissableCubes (end+1:end+4) =  remove_index_numbers;
                        NotAdmissableCubes = unique(NotAdmissableCubes);
                        % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                        % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                        checkcodeNumber=checkcodeNumber+1;
    
                    end
    
                end
            elseif page == 1 || page == size(design_check,1)
                % disp('XZ - 4 voxels');

                rows = [row, size(design_check,1)+1-row, row, size(design_check,1)+1-row];
                columns = [column, column, column, column];
                pages = [page, page, size(design_check,1)+1-page, size(design_check,1)+1-page];

                design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

                % design_check(row, column, page) = 0;
                % design_check(size(design_check,1)+1-row, column, page) = 0;
                % design_check(row, column, size(design_check,1)+1-page) = 0;
                % design_check(size(design_check,1)+1-row, column, size(design_check,1)+1-page) = 0;
    
                [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
                if length(group_data_new.connected_group) == (length(group_data.connected_group)-4)
                    % disp('If')
                    connectivity_check = false;

                    pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                    % pfc.design(row, column, page) = 0;
                    % pfc.design(size(pfc.design,1)+1-row, column, page) = 0;
                    % pfc.design(row, column, size(pfc.design,1)+1-page) = 0;
                    % pfc.design(size(pfc.design,1)+1-row, column, size(pfc.design,1)+1-page) = 0;
    
                    nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5];
                        Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5] ];
    
                    Voxels = Voxels_new;
                    group_data = group_data_new;
    
                    Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                    CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-3:end,:),'rows');
    
                else
                    % disp('Else')
    
                    remove_index_idxs = [
                        row, column, page;
                        size(design_check,1) + 1 - row, column, page;
    
                        row, column, size(design_check,3) + 1 - page;
    
                        size(design_check,1) + 1 - row, column, size(design_check,3) + 1 - page;
    
                        ];
    
                    % Calculate the reverse indices
                    remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                    % remove_voxelss = voxels_check(remove_index_numbers);
    
                    NotAdmissableCubes (end+1:end+4) =  remove_index_numbers;
                    NotAdmissableCubes = unique(NotAdmissableCubes);
                    % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                    % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                    checkcodeNumber=checkcodeNumber+1;
    
                end
    
            else
                % disp('X face - 2 voxels');

                rows = [row, size(design_check,1)+1-row];
                columns = [column, column];
                pages = [page, page];

                design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

                % design_check(row, column, page) = 0;
                % design_check(size(design_check,1)+1-row, column, page) = 0;
    
                [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
                if length(group_data_new.connected_group) == (length(group_data.connected_group)-2)
                    % disp('If')
                    connectivity_check = false;

                    pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                    % pfc.design(row, column, page) = 0;
                    % pfc.design(size(pfc.design,1)+1-row, column, page) = 0;
    
                    nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[size(design_check,1)+1-row-1+0.5, column-1+0.5, page-1+0.5] ];
    
                    Voxels = Voxels_new;
                    group_data = group_data_new;
    
                    Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                    CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-1:end,:),'rows');
    
                else
                    % disp('Else')
    
                    remove_index_idxs = [
                        row, column, page;
                        size(design_check,1) + 1 - row, column, page];
    
                    % Calculate the reverse indices
                    remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                    % remove_voxelss = voxels_check(remove_index_numbers);
    
                    NotAdmissableCubes (end+1:end+2) =  remove_index_numbers;
                    NotAdmissableCubes = unique(NotAdmissableCubes);
                    % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                    % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                    checkcodeNumber=checkcodeNumber+1;
    
                end
    
            end
        elseif column == 1 || column == size(design_check,1)
            if page == 1 || page == size(design_check,1)
                % disp('YZ edge - 4 voxels');
                rows = [row, row, row, row];
                columns = [column, size(design_check,1)+1-column, column, size(design_check,1)+1-column];
                pages = [page, page, size(design_check,1)+1-page, size(design_check,1)+1-page];

                design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

                % design_check(row, column, page) = 0;
                % design_check(row, size(design_check,1)+1-column, page) = 0;
                % design_check(row, column, size(design_check,1)+1-page) = 0;
                % design_check(row, size(design_check,1)+1-column, size(design_check,1)+1-page) = 0;
    
                [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
                if length(group_data_new.connected_group) == (length(group_data.connected_group)-4)
                    % disp('If')
                    connectivity_check = false;

                    pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                    % pfc.design(row, column, page) = 0;
                    % pfc.design(row, size(pfc.design,1)+1-column, page) = 0;
                    % pfc.design(row, column, size(pfc.design,1)+1-page) = 0;
                    % pfc.design(row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 0;
    
                    nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5];
                        Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, size(design_check,1)+1-page-1+0.5] ];
    
                    Voxels = Voxels_new;
                    group_data = group_data_new;
    
                    Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                    CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-3:end,:),'rows');
    
                else
                    % disp('Else')
    
                    remove_index_idxs = [
                        row, column, page;
                        row, size(design_check,2) + 1 - column, page;
                        row, column, size(design_check,3) + 1 - page;
                        row, size(design_check,2) + 1 - column, size(design_check,3) + 1 - page];
    
                    % Calculate the reverse indices
                    remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                    % remove_voxelss = voxels_check(remove_index_numbers);
    
                    NotAdmissableCubes (end+1:end+4) =  remove_index_numbers;
                    NotAdmissableCubes = unique(NotAdmissableCubes);
                    % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                    % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                    checkcodeNumber=checkcodeNumber+1;
    
                end
            else
                % disp('Y face - 2 voxels');

                rows = [row, row];
                columns = [column, size(design_check,1)+1-column];
                pages = [page, page];

                design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

                % design_check(row, column, page) = 0;
                % design_check(row, size(design_check,1)+1-column, page) = 0;
    
                [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
                if length(group_data_new.connected_group) == (length(group_data.connected_group)-2)
                    % disp('If')
                    connectivity_check = false;

                    pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                    % pfc.design(row, column, page) = 0;
                    % pfc.design(row, size(pfc.design,1)+1-column, page) = 0;
    
                    nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                        Size/GridNumber.*[row-1+0.5, size(design_check,1)+1-column-1+0.5, page-1+0.5]];
    
                    Voxels = Voxels_new;
                    group_data = group_data_new;
    
                    Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                    CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-1:end,:),'rows');
    
                else
                    % disp('Else')
    
                    remove_index_idxs = [
                        row, column, page;
                        row, size(design_check,2) + 1 - column, page];
    
                    % Calculate the reverse indices
                    remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                    % remove_voxelss = voxels_check(remove_index_numbers);
                    % NotAdmissableCubes (end+1:end+2) =  remove_voxelss;
                    NotAdmissableCubes (end+1:end+2) =  remove_index_numbers;
                    NotAdmissableCubes = unique(NotAdmissableCubes);
                    % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                    % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                    checkcodeNumber=checkcodeNumber+1;
    
                end
    
            end
        elseif page == 1 || page == size(design_check,1)
            % disp('Z face - 2 voxels');

            rows = [row, row];
            columns = [column, column];
            pages = [page, size(design_check,1)+1-page];

            design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

            % design_check(row, column, page) = 0;
            % design_check(row, column, size(design_check,1)+1-page) = 0;
    
            [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
            if length(group_data_new.connected_group) == (length(group_data.connected_group)-2)
                % disp('If')
                connectivity_check = false;

                pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;

                % pfc.design(row, column, page) = 0;
                % pfc.design(row, column, size(pfc.design,1)+1-page) = 0;
    
                nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[row-1+0.5, column-1+0.5, size(design_check,1)+1-page-1+0.5] ];
    
                Voxels = Voxels_new;
                group_data = group_data_new;
    
                Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end-1:end,:),'rows');
    
            else
                % disp('Else')
    
                remove_index_idxs = [
                    row, column, page;
                    row, column, size(design_check,3) + 1 - page];
    
                % Calculate the reverse indices
                remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                % remove_voxelss = voxels_check(remove_index_numbers);
    
                NotAdmissableCubes (end+1:end+2) =  remove_index_numbers;
                NotAdmissableCubes = unique(NotAdmissableCubes);
                % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                checkcodeNumber=checkcodeNumber+1;
    
            end
    
        else
            % disp('Inside - 1 voxel');

            rows = [row];
            columns = [column];
            pages = [page];

            design_check(sub2ind(size(design_check), rows, columns, pages)) = 0;

            % design_check(row, column, page) = 0;
    
            [group_data_new, isConnected, Voxels_new]  = find_connected_group(design_check);
    
    
            if length(group_data_new.connected_group) == (length(group_data.connected_group)-1)
                % disp('If')
                connectivity_check = false;
                pfc.design(sub2ind(size(design_check), rows, columns, pages)) = 0;
                % pfc.design(row, column, page) = 0;
    
                nodes_to_remove = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5] ];
    
                Voxels = Voxels_new;
                group_data = group_data_new;
    
                Removed_CenterNodes = [Removed_CenterNodes; nodes_to_remove];
                CenterNodes = setdiff(CenterNodes, Removed_CenterNodes(end,:),'rows');
    
            else
                % disp('Else')
    
                remove_index_idxs = [
                    row, column, page];
    
                % Calculate the reverse indices
                remove_index_numbers = sub2ind(size(design_check), remove_index_idxs(:, 1), remove_index_idxs(:, 2), remove_index_idxs(:, 3));
    
                % remove_voxelss = voxels_check(remove_index_numbers);
    
                NotAdmissableCubes (end+1) =  remove_index_numbers;
                NotAdmissableCubes = unique(NotAdmissableCubes);
                % name_of_geom_check=strcat('removeCube',num2str(rng_seed_offset), '_episod', num2str(i_episode_number), 'ActionNumber', num2str(checkcodeNumber));
                % [Slops, Impedances, VolumeFractions] = Cube_Solid_Eigenfrequency (CenterNodesCheck, name_of_geom_check, GridNumber, Size)
                checkcodeNumber=checkcodeNumber+1;
    
            end
        end
    
    end
    
    popc2.pfc=pfc;
    popc2.Voxels=Voxels;
    popc2.group_data=group_data;
    popc2.CenterNodes=CenterNodes;
    
    RFlag = true;
    disp(length(Voxels))
    
end