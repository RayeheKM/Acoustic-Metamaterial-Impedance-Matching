function [popc1, AFlag] = Add(p1, GridNumber, Size)
    
    CenterNodes = p1.CenterNodes;
    group_data = p1.group_data;
    Voxels = p1.Voxels;
    pfc = p1.pfc;
    
    
    Added_CenterNodes = [];
    disp(length(Voxels))
    % disp(length(group_data.all_empty_neighbors));
    if isempty(group_data.all_empty_neighbors)
        disp('Cube is full')
        disp(length(group_data.connected_group))
        % popc1 = Remove(p1, GridNumber, Size);
        popc1 = p1;
        AFlag = false;
        return;
    end
    add_index_idx = randi(length(group_data.all_empty_neighbors));
    add_index = group_data.all_empty_neighbors(add_index_idx);
    [row, column, page] = ind2sub(size(pfc.design), add_index);
    
    if row == 1 || row == size(pfc.design,1)
        if column == 1 || column == size(pfc.design,1)
            if page == 1 || page == size(pfc.design,1)
                % disp('Corner - 8 voxels');
                rows = [row, size(pfc.design,1)+1-row, row, row, size(pfc.design,1)+1-row, size(pfc.design,1)+1-row, row, size(pfc.design,1)+1-row];
                columns = [column, column, size(pfc.design,1)+1-column, column, size(pfc.design,1)+1-column, column, size(pfc.design,1)+1-column, size(pfc.design,1)+1-column];
                pages = [page, page, page, size(pfc.design,1)+1-page, page, size(pfc.design,1)+1-page, size(pfc.design,1)+1-page, size(pfc.design,1)+1-page];

                pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

                % pfc.design(row, column, page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, column, page) = 1;
                % pfc.design(row, size(pfc.design,1)+1-column, page) = 1;
                % pfc.design(row, column, size(pfc.design,1)+1-page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, column, size(pfc.design,1)+1-page) = 1;
                % pfc.design(row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 1;
    
                nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5];
                    Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, size(pfc.design,1)+1-page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, size(pfc.design,1)+1-column-1+0.5, size(pfc.design,1)+1-page-1+0.5] ];
    
                Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
                CenterNodes = [CenterNodes; nodes_to_add];
    
            else
                % disp('XY edge - 4 voxels');

                rows = [row, size(pfc.design,1)+1-row, row, size(pfc.design,1)+1-row];
                columns = [column, column, size(pfc.design,1)+1-column, size(pfc.design,1)+1-column];
                pages = [page, page, page, page];

                pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;
                % pfc.design(row, column, page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, column, page) = 1;
                % pfc.design(row, size(pfc.design,1)+1-column, page) = 1;
                % pfc.design(size(pfc.design,1)+1-row, size(pfc.design,1)+1-column, page) = 1;
    
                nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5];
                    Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5]];
    
                Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
                CenterNodes = [CenterNodes; nodes_to_add];
    
            end
        elseif page == 1 || page == size(pfc.design,1)
            % disp('XZ - 4 voxels');

            rows = [row, size(pfc.design,1)+1-row, row, size(pfc.design,1)+1-row];
            columns = [column, column, column, column];
            pages = [page, page, size(pfc.design,1)+1-page, size(pfc.design,1)+1-page];

            pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

            % pfc.design(row, column, page) = 1;
            % pfc.design(size(pfc.design,1)+1-row, column, page) = 1;
            % pfc.design(row, column, size(pfc.design,1)+1-page) = 1;
            % pfc.design(size(pfc.design,1)+1-row, column, size(pfc.design,1)+1-page) = 1;
    
            nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, page-1+0.5];
                Size/GridNumber.*[row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5];
                Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5] ];
    
            Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
            CenterNodes = [CenterNodes; nodes_to_add];
        else
            % disp('X face - 2 voxels');

            rows = [row, size(pfc.design,1)+1-row];
            columns = [column, column];
            pages = [page, page];

            pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

            % pfc.design(row, column, page) = 1;
            % pfc.design(size(pfc.design,1)+1-row, column, page) = 1;
    
            nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                Size/GridNumber.*[size(pfc.design,1)+1-row-1+0.5, column-1+0.5, page-1+0.5] ];
    
            Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
            CenterNodes = [CenterNodes; nodes_to_add];
    
        end
    elseif column == 1 || column == size(pfc.design,1)
        if page == 1 || page == size(pfc.design,1)
            % disp('YZ edge - 4 voxels');

            rows = [row, row, row, row];
            columns = [column, size(pfc.design,1)+1-column, column, size(pfc.design,1)+1-column];
            pages = [page, page, size(pfc.design,1)+1-page, size(pfc.design,1)+1-page];

            pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

            % pfc.design(row, column, page) = 1;
            % pfc.design(row, size(pfc.design,1)+1-column, page) = 1;
            % pfc.design(row, column, size(pfc.design,1)+1-page) = 1;
            % pfc.design(row, size(pfc.design,1)+1-column, size(pfc.design,1)+1-page) = 1;
    
            nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5];
                Size/GridNumber.*[row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5];
                Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, size(pfc.design,1)+1-page-1+0.5] ];
    
            Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
            CenterNodes = [CenterNodes; nodes_to_add];
        else
            % disp('Y face - 2 voxels');

            rows = [row, row];
            columns = [column, size(pfc.design,1)+1-column];
            pages = [page, page];

            pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

            % pfc.design(row, column, page) = 1;
            % pfc.design(row, size(pfc.design,1)+1-column, page) = 1;
    
            nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
                Size/GridNumber.*[row-1+0.5, size(pfc.design,1)+1-column-1+0.5, page-1+0.5]];
    
            Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
            CenterNodes = [CenterNodes; nodes_to_add];
    
        end
    elseif page == 1 || page == size(pfc.design,1)
        % disp('Z face - 2 voxels');

        rows = [row, row];
        columns = [column, column];
        pages = [page, size(pfc.design,1)+1-page];

        pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;
        
        % pfc.design(row, column, page) = 1;
        % pfc.design(row, column, size(pfc.design,1)+1-page) = 1;
    
        nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5];
            Size/GridNumber.*[row-1+0.5, column-1+0.5, size(pfc.design,1)+1-page-1+0.5] ];
    
        Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
        CenterNodes = [CenterNodes; nodes_to_add];
    else
        % disp('Inside - 1 voxel');

        rows = [row];
        columns = [column];
        pages = [page];

        pfc.design(sub2ind(size(pfc.design), rows, columns, pages)) = 1;

        % pfc.design(row, column, page) = 1;
    
        nodes_to_add = [Size/GridNumber.*[row-1+0.5, column-1+0.5, page-1+0.5] ];
    
        Added_CenterNodes = [Added_CenterNodes; nodes_to_add];
        CenterNodes = [CenterNodes; nodes_to_add];
    end
    
    [group_data, isConnected, Voxels]  = find_connected_group(pfc.design);
    
    popc1.pfc=pfc;
    popc1.Voxels=Voxels;
    popc1.group_data=group_data;
    popc1.CenterNodes=CenterNodes;

    AFlag = true;

    disp(length(Voxels))
    
end