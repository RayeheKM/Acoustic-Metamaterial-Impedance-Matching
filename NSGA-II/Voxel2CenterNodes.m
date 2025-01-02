function CenterNodes = Voxel2CenterNodes(voxel_design)

GridNumber = 10;
Size = 1;

% Find the indices of the material voxels
[a, b, c] = ind2sub(size(voxel_design), find(voxel_design));

% Calculate the center nodes for each material voxel
CenterNodes = zeros(length(a), 3);
for j = 1:length(a)
    CenterNodes(j, :) = Size / GridNumber .* [a(j) - 1 + 0.5, b(j) - 1 + 0.5, c(j) - 1 + 0.5];
end

