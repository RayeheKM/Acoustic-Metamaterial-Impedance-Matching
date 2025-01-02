clc;
clear;
% close all;

% addpath('/shared/Apps/COMSOL61/multiphysics/mli') %On the server
% addpath('/usr/local/comsol61/multiphysics/mli') %On the server
% mphstart(12111); %On the server
% Directory = '/home/rkm41/NSGA2/';

Directory_files='C:\Users\BrinsonLab\Desktop\Modeling Random Metamaterial\';
Directory = 'C:\Users\BrinsonLab\Desktop\Project files\Alex files\design generation\';
Directory2 = 'C:\Users\BrinsonLab\Desktop\Modeling Random Metamaterial\Periodic Cube Codes\';
pathsToAdd = {Directory_files, Directory, Directory2};
addpath(pathsToAdd{:})

%% NSGA-II Parameters

MaxIt=2;      % Maximum Number of Iterations

nPop=2;        % Population Size

pCrossover=1;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)

% pMutation=0.4;                          % Mutation Percentage
% nMutation=round(pMutation*nPop);        % Number of Mutants
% 
% mu=0.02;                    % Mutation Rate
% 
% sigma=0.1*(VarMax-VarMin);  % Mutation Step Size

%% Initialization

empty_individual.pfc = [];
empty_individual.CenterNodes = [];
% empty_individual.BoundaryCheck = [];
empty_individual.Voxels = [];
empty_individual.group_data = [];
empty_individual.Slop = [];
empty_individual.Impedance = [];
empty_individual.VolumeFraction = [];
empty_individual.Cost = [];
% empty_individual.Position=[];
% empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.CrowdingDistance=[];

pop=repmat(empty_individual,nPop,1);

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
    [Impedance_Static3, Impedance_Dynamic3, Stiffness3, VolumeFraction] = Cube_Static_Eigenfrequency (CenterNodes, name_of_geom, GridNumber, Size);
    % Impedance_Static3=10^6*rand();
    % Impedance_Dynamic3=10^6*rand();
    % Stiffness3=10^6*rand();
    % VolumeFraction=rand();

    % Assign the parameters to the individual
    pop(i).pfc = pfc;
    pop(i).CenterNodes = CenterNodes;
    % pop(i).BoundaryCheck = BoundaryCheck;
    pop(i).Voxels = Voxels;
    pop(i).group_data = group_data;

    % Assign the parameters to the individual
    pop(i).Impedance_Static3 = Impedance_Static3;
    pop(i).Impedance_Dynamic3 = Impedance_Dynamic3;
    pop(i).Stiffness3 = Stiffness3;
    pop(i).VolumeFraction = VolumeFraction;

    % Cost would be the difference between our impedance and water impedance
    % pop(i).Cost=CostFunction(pop(i).Position);
    pop(i).Cost = [abs(pop(i).Impedance_Dynamic3 - TargetImpedance); pop(i).Stiffness3];
    
end

% Non-Dominated Sorting
[pop, F]=NonDominatedSorting(pop);

% Calculate Crowding Distance
pop=CalcCrowdingDistance(pop,F);

% Sort Population
[pop, F]=SortPopulation(pop);

vars_to_save = {'pop', 'F'};
save('NSGA2_InitialPopulation.mat',vars_to_save{:})

%% NSGA-II Main Loop

for it=1:MaxIt
    
    % Crossover
    popc=repmat(empty_individual,nCrossover,1);
    for k=1:nCrossover/2

        disp(['k=',num2str(k)])

        % Select First Parent
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        % Select Second Parent
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        % Perform Crossover
        % [popc1, popc2] = ArithmeticCrossover(p1,p2, GridNumber, Size);
        popc1 = ArithmeticAdd(p1, GridNumber, Size);
        popc2 = ArithmeticRemove(p2, GridNumber, Size);
            
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
        [Impedance_Static3, Impedance_Dynamic3, Stiffness3, VolumeFraction] = Cube_Static_Eigenfrequency (popc1.CenterNodes, name_of_geom, GridNumber, Size);
        % Impedance_Static3=10^6*rand();
        % Impedance_Dynamic3=10^6*rand();
        % Stiffness3=10^6*rand();
        % VolumeFraction=rand();

        % Assign the parameters to the individual
        popc(2*k-1).Impedance_Static3 = Impedance_Static3;
        popc(2*k-1).Impedance_Dynamic3 = Impedance_Dynamic3;
        popc(2*k-1).Stiffness3 = Stiffness3;
        popc(2*k-1).VolumeFraction = VolumeFraction;

        % Cost would be the difference between our impedance and water impedance
        popc(2*k-1).Cost = [abs(popc(2*k-1).Impedance_Dynamic3 - TargetImpedance); popc(2*k-1).Stiffness3];

        % Evaluate Offspring 2
        name_of_geom = [NameGeom,'Crossover',num2str(2*k)];
        [Impedance_Static3, Impedance_Dynamic3, Stiffness3, VolumeFraction] = Cube_Static_Eigenfrequency (popc2.CenterNodes, name_of_geom, GridNumber, Size);
        % Impedance_Static3=10^6*rand();
        % Impedance_Dynamic3=10^6*rand();
        % Stiffness3=10^6*rand();
        % VolumeFraction=rand();
        
        % Assign the parameters to the individual
        popc(2*k).Impedance_Static3 = Impedance_Static3;
        popc(2*k).Impedance_Dynamic3 = Impedance_Dynamic3;
        popc(2*k).Stiffness3 = Stiffness3;
        popc(2*k).VolumeFraction = VolumeFraction;

        % Cost would be the difference between our impedance and water impedance
        popc(2*k).Cost = [abs(popc(2*k).Impedance_Dynamic3 - TargetImpedance); popc(2*k).Stiffness3];
        
    end
    % popc=popc(:);
    % 
    % % Mutation
    % popm=repmat(empty_individual,nMutation,1);
    % for k=1:nMutation
    % 
    %     i=randi([1 nPop]);
    %     p=pop(i);
    % 
    %     popm(k).Position=Mutate(p.Position,mu,sigma);
    % 
    %     popm(k).Cost=CostFunction(popm(k).Position);
    % 
    % end
    
    % Merge
    pop=[pop
         popc];
     
    % Non-Dominated Sorting
    [pop, F]=NonDominatedSorting(pop);

    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);

    % Sort Population
    [pop, F]=SortPopulation(pop);
    
    % Truncate
    pop=pop(1:nPop);
    
    % Non-Dominated Sorting
    [pop, F]=NonDominatedSorting(pop);

    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);

    % Sort Population
    [pop, F]=SortPopulation(pop);
    
    % Store F1
    F1=pop(F{1});
    
    vars_to_save = {'pop', 'F', 'F1'};
    save(['NSGA2_Population_',num2str(it),'.mat'],vars_to_save{:})

    % % Show Iteration Information
    % disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);
    % 
    % % Plot F1 Costs
    % figure(1);
    % PlotCosts(F1);
    
end

