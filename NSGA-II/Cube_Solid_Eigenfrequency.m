function [Slop, Impedance, VolumeFraction] = Cube_Solid_Eigenfrequency (CenterNodes, NameData, GridNumber, Size)
%%
% Metamaterial_Shell.m
%
% Model exported on Feb 20 2023, 09:51 by COMSOL 6.1.0.282.
% addpath('/usr/local/comsol61/multiphysics/mli');
% mphstart(12343);

% clc
% clear all

% tic
Directory_files='/home/rkm41/SAGA/';
% Directory_files='C:\Users\BrinsonLab\Desktop\Modeling Random Metamaterial\';
name_of_model=strcat(NameData,'Cube.mph');
model_filename = [Directory_files,name_of_model];

StepSize=10;
Number_of_Eigenfrequencies = 3;
% Target frequency
targetFrequency = 250e3; % 250 kHz

Number_of_Cubes=length(CenterNodes);

import com.comsol.model.*
import com.comsol.model.util.* 

model = ModelUtil.create('Model');

Young=3.3; %GPa
% Young=0.1:20:200.1;
rho= 1190; % kg/m3
% rho=900:1000:8900;

model.component.create('comp1', true);

CellSize=Size; %mm
Grid_Number=GridNumber;
uz=0.5; %mm

model.param.set('CellSize', [num2str(CellSize),' [mm]']);
model.param.set('Grid_Number', [num2str(Grid_Number)]);
model.param.set('uz', [num2str(uz),' [mm]']);
model.param.set('kx', '0 [rad/m]');
model.param.set('ky', '0 [rad/m]');
model.param.set('kz', '0 [rad/m]');

model.component('comp1').geom.create('geom1', 3);

model.component('comp1').geom('geom1').lengthUnit('mm');

% disp(['number of cubes:',num2str(Number_of_Cubes)])

for i=1:Number_of_Cubes
    
    model.component('comp1').geom('geom1').create(['blk',num2str(i)], 'Block');
    model.component('comp1').geom('geom1').feature(['blk',num2str(i)]).set('size', {'CellSize/Grid_Number' 'CellSize/Grid_Number' 'CellSize/Grid_Number'});
    model.component('comp1').geom('geom1').feature(['blk',num2str(i)]).set('base', 'center');
    model.component('comp1').geom('geom1').feature(['blk',num2str(i)]).set('pos', CenterNodes(i,:));
    
end

model.component('comp1').geom('geom1').run;
% model.component('comp1').geom('geom1').run('fin');

% disp(model_filename)
% model.save(model_filename)

model.component('comp1').material.create('mat1', 'Common');
model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');

model.component('comp1').massProp.create('mass1', 'MassProperties');
model.component('comp1').massProp('mass1').selection.geom('geom1', 3);
model.component('comp1').massProp('mass1').selection.all;
model.component('comp1').massProp('mass1').set('densitySource', 'fromPhysics');

model.component('comp1').material('mat1').propertyGroup('Enu').set('E', [num2str(Young),' [GPa]']);
model.component('comp1').material('mat1').propertyGroup('def').set('density', [num2str(rho),' [kg/m^3]']);
model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', '0.39');

% model.save(model_filename)

model.component('comp1').selection.create('box1', 'Box');
model.component('comp1').selection('box1').label('ZN');
model.component('comp1').selection('box1').set('entitydim', 2);
model.component('comp1').selection('box1').set('xmin', 0);
model.component('comp1').selection('box1').set('xmax', 'CellSize');
model.component('comp1').selection('box1').set('ymin', 0);
model.component('comp1').selection('box1').set('ymax', 'CellSize');
model.component('comp1').selection('box1').set('zmin', 0);
model.component('comp1').selection('box1').set('zmax', 0);
model.component('comp1').selection('box1').set('condition', 'inside');

model.component('comp1').selection.create('box2', 'Box');
model.component('comp1').selection('box2').label('ZP');
model.component('comp1').selection('box2').set('entitydim', 2);
model.component('comp1').selection('box2').set('xmin', 0);
model.component('comp1').selection('box2').set('xmax', 'CellSize');
model.component('comp1').selection('box2').set('ymin', 0);
model.component('comp1').selection('box2').set('ymax', 'CellSize');
model.component('comp1').selection('box2').set('zmin', 'CellSize');
model.component('comp1').selection('box2').set('zmax', 'CellSize');
model.component('comp1').selection('box2').set('condition', 'inside');

model.component('comp1').selection.create('box3', 'Box');
model.component('comp1').selection('box3').label('XN');
model.component('comp1').selection('box3').set('entitydim', 2);
model.component('comp1').selection('box3').set('xmin', 0);
model.component('comp1').selection('box3').set('xmax', 0);
model.component('comp1').selection('box3').set('ymin', 0);
model.component('comp1').selection('box3').set('ymax', 'CellSize');
model.component('comp1').selection('box3').set('zmin', 0);
model.component('comp1').selection('box3').set('zmax', 'CellSize');
model.component('comp1').selection('box3').set('condition', 'inside');

model.component('comp1').selection.create('box4', 'Box');
model.component('comp1').selection('box4').label('XP');
model.component('comp1').selection('box4').set('entitydim', 2);
model.component('comp1').selection('box4').set('xmin', 'CellSize');
model.component('comp1').selection('box4').set('xmax', 'CellSize');
model.component('comp1').selection('box4').set('ymin', 0);
model.component('comp1').selection('box4').set('ymax', 'CellSize');
model.component('comp1').selection('box4').set('zmin', 0);
model.component('comp1').selection('box4').set('zmax', 'CellSize');
model.component('comp1').selection('box4').set('condition', 'inside');
% 
model.component('comp1').selection.create('box5', 'Box');
model.component('comp1').selection('box5').label('YN');
model.component('comp1').selection('box5').set('entitydim', 2);
model.component('comp1').selection('box5').set('xmin', 0);
model.component('comp1').selection('box5').set('xmax', 'CellSize');
model.component('comp1').selection('box5').set('ymin', 0);
model.component('comp1').selection('box5').set('ymax', 0);
model.component('comp1').selection('box5').set('zmin', 0);
model.component('comp1').selection('box5').set('zmax', 'CellSize');
model.component('comp1').selection('box5').set('condition', 'inside');

model.component('comp1').selection.create('box6', 'Box');
model.component('comp1').selection('box6').label('YP');
model.component('comp1').selection('box6').set('entitydim', 2);
model.component('comp1').selection('box6').set('xmin', 0);
model.component('comp1').selection('box6').set('xmax', 'CellSize');
model.component('comp1').selection('box6').set('ymin', 'CellSize');
model.component('comp1').selection('box6').set('ymax', 'CellSize');
model.component('comp1').selection('box6').set('zmin', 0);
model.component('comp1').selection('box6').set('zmax', 'CellSize');
model.component('comp1').selection('box6').set('condition', 'inside');

% model.component('comp1').selection.create('ball1', 'Ball');
% model.component('comp1').selection('ball1').set('entitydim', 0);
% model.component('comp1').selection('ball1').set('condition', 'inside');
% model.component('comp1').selection('ball1').label('origin');
% model.component('comp1').selection('ball1').set('posx', '0');
% model.component('comp1').selection('ball1').set('posy', '0');
% model.component('comp1').selection('ball1').set('posz', '0');
% model.component('comp1').selection('ball1').set('r', 'CellSize/10000');

model.component('comp1').selection.create('uni1', 'Union');
model.component('comp1').selection('uni1').set('entitydim', 2);
model.component('comp1').selection('uni1').label('Union Z');
model.component('comp1').selection('uni1').set('input', {'box1' 'box2'});

model.component('comp1').selection.create('uni2', 'Union');
model.component('comp1').selection('uni2').set('entitydim', 2);
model.component('comp1').selection('uni2').label('Union X');
model.component('comp1').selection('uni2').set('input', {'box3' 'box4'});

model.component('comp1').selection.create('uni3', 'Union');
model.component('comp1').selection('uni3').set('entitydim', 2);
model.component('comp1').selection('uni3').label('Union Y');
model.component('comp1').selection('uni3').set('input', {'box5' 'box6'});

% model.save(model_filename)

model.component('comp1').physics.create('solid', 'SolidMechanics', 'geom1');
model.component('comp1').physics('solid').prop('ShapeProperty').set('order_displacement', 1);
model.component('comp1').physics('solid').create('pc1', 'PeriodicCondition', 2);
model.component('comp1').physics('solid').feature('pc1').selection.named('uni1');
model.component('comp1').physics('solid').feature('pc1').set('PeriodicType', 'Floquet');
model.component('comp1').physics('solid').feature('pc1').set('kFloquet', {'kx'; 'ky'; 'kz'});
model.component('comp1').physics('solid').create('pc2', 'PeriodicCondition', 2);
model.component('comp1').physics('solid').feature('pc2').selection.named('uni2');
model.component('comp1').physics('solid').feature('pc2').set('PeriodicType', 'Floquet');
model.component('comp1').physics('solid').feature('pc2').set('kFloquet', {'kx'; 'ky'; 'kz'});
model.component('comp1').physics('solid').create('pc3', 'PeriodicCondition', 2);
model.component('comp1').physics('solid').feature('pc3').selection.named('uni3');
model.component('comp1').physics('solid').feature('pc3').set('PeriodicType', 'Floquet');
model.component('comp1').physics('solid').feature('pc3').set('kFloquet', {'kx'; 'ky'; 'kz'});

% model.save(model_filename)

model.component('comp1').mesh.create('mesh1');
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 3);
model.component('comp1').mesh('mesh1').create('map1', 'Map');
model.component('comp1').mesh('mesh1').feature('map1').selection.named('uni1');
model.component('comp1').mesh('mesh1').feature('map1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('map1').feature('size1').set('hauto', 2);
model.component('comp1').mesh('mesh1').create('swe1', 'Sweep');
model.component('comp1').mesh('mesh1').feature('swe1').create('size1', 'Size');
model.component('comp1').mesh('mesh1').feature('swe1').feature('size1').set('hauto', 2);
model.component('comp1').mesh('mesh1').run;

% model.save(model_filename)

% model.study.create('std1');
% model.study('std1').create('param', 'Parametric');
% model.study('std1').create('eig', 'Eigenfrequency');
% model.study('std1').feature('param').set('pname', {'kz'});
% % model.study('std1').feature('param').set('plistarr', {'0, pi/CellSize/200'});
% model.study('std1').feature('param').set('plistarr', {['range(0,pi/CellSize/',num2str(StepSize),',pi/CellSize)']});
% model.study('std1').feature('param').set('punit', {'rad/m'});
% model.study('std1').feature('eig').set('neigs', 3);
% model.study('std1').feature('eig').set('neigsactive', true);
% model.study('std1').feature('eig').set('shiftactive', false);

model.study.create('std1');
model.study('std1').create('eig', 'Eigenfrequency');
% model.study('std1').feature('eig').set('plotgroup', 'Default');
% model.study('std1').feature('eig').set('conrad', '1');
% model.study('std1').feature('eig').set('solnum', 'auto');
% model.study('std1').feature('eig').set('notsolnum', 'auto');
% model.study('std1').feature('eig').set('ngenAUX', '1');
% model.study('std1').feature('eig').set('goalngenAUX', '1');
% model.study('std1').feature('eig').set('ngenAUX', '1');
% model.study('std1').feature('eig').set('goalngenAUX', '1');
% model.study('std1').feature('eig').setSolveFor('/physics/solid', true);

% model.component('comp1').common.create('mpf1', 'ParticipationFactors');

model.study('std1').feature('eig').set('neigsactive', true);
model.study('std1').feature('eig').set('neigs', Number_of_Eigenfrequencies);
model.study('std1').feature('eig').set('shiftactive', false);
model.study('std1').feature('eig').set('useparam', true);
model.study('std1').feature('eig').setIndex('pname', 'CellSize', 0);
model.study('std1').feature('eig').setIndex('plistarr', '', 0);
model.study('std1').feature('eig').setIndex('punit', 'm', 0);
model.study('std1').feature('eig').setIndex('pname', 'CellSize', 0);
model.study('std1').feature('eig').setIndex('plistarr', '', 0);
model.study('std1').feature('eig').setIndex('punit', 'm', 0);
model.study('std1').feature('eig').setIndex('pname', 'kz', 0);
model.study('std1').feature('eig').setIndex('plistarr', 'range(0,pi/CellSize/10,pi/CellSize)', 0);


model.save(model_filename)
% Slop=0;
% Impedance=0;
% VolumeFraction=0;

model.study('std1').run

% model.save(model_filename)

% Fr1=mphglobal(model,'solid.freq', 'data', 'dset2');
% Fr2=mphglobal(model,'solid.freq', 'data', 'dset1');

Frqs = mphglobal(model,'solid.freq');
kzs = mphglobal(model,'kz');

% FRAll = mphglobal(model,'solid.freq', 'data', 'dset2', 'outersolnum', 'all');
% si = mphsolinfo(model);
% eigenvalues = (si.solvals/(-1i*2*pi));  %lambda
% Slop=real(FRAll(3,2)-FRAll(3,1))*2*pi/(pi/(CellSize/1000)/200);
% frequencies = FRAll(3,:);
% kz=linspace(0,pi/(CellSize/1000),StepSize+1);
frequencies = Frqs (Number_of_Eigenfrequencies:Number_of_Eigenfrequencies:end);
kz=kzs(Number_of_Eigenfrequencies:Number_of_Eigenfrequencies:end);

nearestFreq2 = 0;
nearestFreq1 = 0;
% delta_kz = kz(Number_of_Eigenfrequencies+1)-kz(1);

for j = 1:numel(frequencies)-1
    kz_l = kz(j);
    frequencies_l = frequencies(j);
    kz_n = kz(j+1);
    frequencies_n = frequencies(j+1);
    if frequencies_l <= targetFrequency && frequencies_n >= targetFrequency
        nearestFreq2 = frequencies_n;
        nearestFreq1 = frequencies_l;
        delta_kz = kz_n - kz_l;
        Slop = real(nearestFreq2 - nearestFreq1)*2*pi/delta_kz;
        break
    end
end

if nearestFreq1 == 0 && nearestFreq2 == 0
    disp('No intersection')
    Slop = 0;
end

% p = [0 targetFrequency];
% kz=linspace(0,pi/(CellSize/1000),StepSize+1);
% 
% frq_int = polyval(p,kz);

% tolerance = 1e-6; % Set a suitable tolerance
% 
% % Find indices where the difference between frq_int and frequencies is small
% intersection_indices = find(abs(frq_int - frequencies) < tolerance);
% 
% % Retrieve the corresponding kz values
% kz_int = kz(intersection_indices);
% 
% nearestFreq1 = 0;
% nearestFreq2 = 0;
% 
% if numel(kz_int) >= 2
%     % Sort intersection_indices based on corresponding kz_int values
%     [~, sortedIndices] = sort(kz_int);
%     
%     nearestFreq1 = frequencies(intersection_indices(sortedIndices(1)))
%     nearestFreq2 = frequencies(intersection_indices(sortedIndices(2)))
%     
%     if nearestFreq1 > nearestFreq2
%         temp = nearestFreq1;
%         nearestFreq1 = nearestFreq2;
%         nearestFreq2 = temp;
%     end
% end
% 
% 
% disp('nearestFreq1:')
% disp(nearestFreq1)
% 
% disp('nearestFreq2:')
% disp(nearestFreq2)

% kz_int = interp1(real(frq_int-frequencies), kz, 0);
% 
% nearestFreq1=0;
% nearestFreq2=0;
% 
% if ~any(isnan(kz_int))
% 
%     differences = abs(kz - kz_int(1));
%     [~, sortedIndices] = sort(differences);
%     
%     nearestFreq1 = frequencies(sortedIndices(1));
%     nearestFreq2 = frequencies(sortedIndices(2));
%     if nearestFreq1 > nearestFreq2
%         temp = nearestFreq1;
%         nearestFreq1 = nearestFreq2;
%         nearestFreq2 = temp;
%     end
% end
% 
% disp('nearestFreq1:')
% disp(nearestFreq1)
% 
% disp('nearestFreq2:')
% disp(nearestFreq2)
% % Calculate the absolute differences between the target frequency and all frequencies
% differences = abs(frequencies - targetFrequency);
% [~, sortedIndices] = sort(differences);
% 
% nearestFreq1 = frequencies(sortedIndices(1));
% nearestFreq2 = frequencies(sortedIndices(2));
% 
% % To ensure the smaller frequency is stored in nearestFreq1
% if nearestFreq1 > nearestFreq2
%     temp = nearestFreq1;
%     nearestFreq1 = nearestFreq2;
%     nearestFreq2 = temp;
% end
% 
% % To remove the designs that their third dispersion curve does not
% % intersect with f=targetFrequency 
% if nearestFreq1<targetFrequency && nearestFreq2<targetFrequency
%     nearestFreq1=0;
%     nearestFreq2=0;
% end
% 


% Calculate the slope 
% Slop = real(nearestFreq2 - nearestFreq1)*2*pi/(pi/(CellSize/1000)/10);



Mass_data=mpheval(model,'mass1.mass');
MassValue=Mass_data.d1(1);
Total_Volume=(CellSize/1000)^3;
DensityStructure=MassValue/Total_Volume;
Cube_Volume=((CellSize/1000)/(Grid_Number))^3;

Impedance=Slop*DensityStructure;
VolumeFraction = (Number_of_Cubes*Cube_Volume)/Total_Volume;

end