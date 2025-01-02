clc;
clear;
% close all;

% addpath('/shared/Apps/COMSOL61/multiphysics/mli') %On the server
% addpath('/usr/local/comsol61/multiphysics/mli') %On the server
% mphstart(12344); %On the server
% Directory = '/home/rkm41/';

Directory_files='C:\Users\BrinsonLab\Desktop\Modeling Random Metamaterial\';
Directory = 'C:\Users\BrinsonLab\Desktop\Project files\Alex files\design generation\';
pathsToAdd = {Directory_files, Directory};
addpath(pathsToAdd{:})

%% Problem Definition

% CostFunction=@(x) Sphere(x);        % Cost Function

% nVar=5;             % Number of Decision variables

% VarSize=[1 nVar];   % Variables Martix Size

% VarMin=-10;         % Variables Lower Bound
% VarMax= 10;         % Variables Upper Bound
% 
% if numel(VarMin)==1                 % If VarMin is Scalar
%     VarMin=repmat(VarMin,VarSize);  % Convert VarMin to Vector
% end
% 
% if numel(VarMax)==1                 % If VarMax is Scalar
%     VarMax=repmat(VarMax,VarSize);  % Convert VarMax to Vector
% end

%% GA Parameters

MaxIt=50;       % Maximum Number of Iterations

nPop=100;          % Population Size

pc=1;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number fo Parents (Offsprings)

% pm=0.3;                 % Mutation Percentage
% nm=round(pm*nPop);      % Number of Mutants

% gamma=0.2;              % Crossover Inflation Rate

% mu=0.1;                 % Mutation Rate
% 
% MutationMode='rand';    % Mutation Mode

% eta=0.1;                    % Mutation Step Size Ratio
% sigma=eta*(VarMax-VarMin);  % Mutation Step Size

%% Initialization

global NFE;
NFE=0;

% Create Empty Structure
% Create an empty structure
empty_individual.pfc = [];
empty_individual.CenterNodes = [];
% empty_individual.BoundaryCheck = [];
empty_individual.Voxels = [];
empty_individual.group_data = [];
empty_individual.Slop = [];
empty_individual.Impedance = [];
empty_individual.VolumeFraction = [];
empty_individual.Cost = []; % Change this field name to 'cost'

% Create a structure array to save population data
pop = repmat(empty_individual, nPop, 1);


NameData='CubeNew';
NameGeom='designs';

Size=1; %mm
GridNumber=10;

TargetImpedance = 1.48e6;

% Initilize Population
for i=1:nPop
    % Create Random Solution (Position)
    % pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    disp(['i=',num2str(i)])

    name_of_geom = [NameGeom,num2str(i)];

    rng_seed_offset=i;
    GeometryDirectory = [Directory,name_of_geom];
    % Generating a design
    [pfc, CenterNodes, BoundaryCheck, Voxels, group_data] = generate_connected_design(rng_seed_offset,GridNumber, Size);
    % Evaluating the design
    % [Slop, Impedance, VolumeFraction] = Cube_Solid_Eigenfrequency (CenterNodes, name_of_geom, GridNumber, Size);
    Slop = 0;
    Impedance = 10^6*rand();
    VolumeFraction = 0;
    % Assign the parameters to the individual
    pop(i).pfc = pfc;
    pop(i).CenterNodes = CenterNodes;
    % pop(i).BoundaryCheck = BoundaryCheck;
    pop(i).Voxels = Voxels;
    pop(i).group_data = group_data;
    pop(i).Slop = Slop;
    pop(i).Impedance = Impedance;
    pop(i).VolumeFraction = VolumeFraction;

    % Cost would be the difference between our impedance and water impedance
    pop(i).Cost = abs(pop(i).Impedance - TargetImpedance);

end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

% Store Best Solution Ever Found
BestSol=pop(1);
BestCost0=BestSol.Cost;
BestImpedance0=BestSol.Impedance;

vars_to_save = {'pop', 'BestSol', 'BestCost0', 'BestImpedance0'};
save(['InitialPopulation.mat'],vars_to_save{:})

% Create Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
BestImpedance = zeros(MaxIt,1);

% Create Array to Hold NFEs
nfe=zeros(MaxIt,1);


%% GA Main Loop

for it=1:MaxIt
    
    disp(['it=',num2str(it)])
    % Perform Crossover
    popc=repmat(empty_individual,nc,1);
    for k=1:nc/2
        disp(['k=',num2str(k)])

        % Select First Parent
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        % Select Second Parent
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        % Perform Crossover
        [popc1, popc2] = ArithmeticCrossover(p1,p2, GridNumber, Size);
            
        popc(2*k-1).pfc = popc1.pfc;
        popc(2*k-1).CenterNodes = popc1.CenterNodes;
        popc(2*k-1).Voxels = popc1.Voxels;
        popc(2*k-1).group_data = popc1.group_data;
        popc(2*k).pfc = popc2.pfc;
        popc(2*k).CenterNodes = popc2.CenterNodes;
        popc(2*k).Voxels = popc2.Voxels;
        popc(2*k).group_data = popc2.group_data;
        
        % Evaluate Offspring 1
        name_of_geom = [NameGeom,'Crossover',num2str(k)];
        % [Slop, Impedance, VolumeFraction] = Cube_Solid_Eigenfrequency (popc(2*k-1).CenterNodes, name_of_geom, GridNumber, Size);
        Slop = 0;
        Impedance = 10^6*rand();
        VolumeFraction = 0;
        % Assign the parameters to the individual
        popc(2*k-1).Slop = Slop;
        popc(2*k-1).Impedance = Impedance;
        popc(2*k-1).VolumeFraction = VolumeFraction;

        % Cost would be the difference between our impedance and water impedance
        popc(2*k-1).Cost = abs(popc(2*k-1).Impedance - TargetImpedance);

        % Evaluate Offspring 2
        name_of_geom = [NameGeom,'Crossover',num2str(2*k)];
        % [Slop, Impedance, VolumeFraction] = Cube_Solid_Eigenfrequency (popc(2*k).CenterNodes, name_of_geom, GridNumber, Size);
        Slop = 0;
        Impedance = 10^6*rand();
        VolumeFraction = 0;

        % Assign the parameters to the individual
        popc(2*k).Slop = Slop;
        popc(2*k).Impedance = Impedance;
        popc(2*k).VolumeFraction = VolumeFraction;

        % Cost would be the difference between our impedance and water impedance
        popc(2*k).Cost = abs(popc(2*k).Impedance - TargetImpedance);

    end


    % Merge Pops
    pop=[pop
         popc];
    

    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Truncate Extra Individuals
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Store Best Solution Ever Found
    BestSol=pop(1);
    
    % Store Best Cost
    BestCost(it)=BestSol.Cost;
    BestImpedance(it)=BestSol.Impedance;

    % Store NFE
    nfe(it)=NFE;

    vars_to_save = {'pop', 'BestSol', 'BestCost', 'BestImpedance'};
    save(['Population_',num2str(it),'.mat'],vars_to_save{:})

    % Show Iteration Information
    disp(['Iteration ' num2str(it) ...
          ': Best Cost = ' num2str(BestCost(it)) ...
          ', NFE = ' num2str(nfe(it))]);
    
end

%% Results

figure;
semilogy(BestCost,'LineWidth',2);
xlabel('Generation');
ylabel('Best Cost');

figure;
semilogy(BestImpedance,'LineWidth',2);
xlabel('Generation');
ylabel('Best design''s Impedance (Rayl)');

figure;
semilogy(nfe,BestCost,'LineWidth',2);
xlabel('NFE');
ylabel('Best Cost');
