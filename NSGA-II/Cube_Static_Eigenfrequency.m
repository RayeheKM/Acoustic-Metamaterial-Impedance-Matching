function [Impedance_Static3, Impedance_Dynamic3, Stiffness3, VolumeFraction] = Cube_Static_Eigenfrequency (CenterNodes, NameData, GridNumber, Size)

    % Directory_files='/home/rkm41/NSGA2_Max/';
    Directory_files='C:\Users\BrinsonLab\Desktop\Modeling Random Metamaterial\';
    name_of_model=strcat(NameData,'Cube.mph');
    model_filename = [Directory_files,name_of_model];
    
    % StepSize=10;
    Number_of_Eigenfrequencies = 3;
    % Target frequency
    targetFrequency = 250e3; % 250 kHz
    
    Number_of_Cubes=length(CenterNodes);
    
    import com.comsol.model.*
    import com.comsol.model.util.*
    
    model = ModelUtil.create('Model');
    
    Young = 100; %GPa
    rho = 7850; % kg/m3
    nu = 0.39;
    
    model.component.create('comp1', true);
    
    CellSize=Size; %mm
    Grid_Number=GridNumber;
    u_disp=0.1; %mm
    
    model.param.set('CellSize', [num2str(CellSize),' [mm]']);
    model.param.set('Grid_Number', [num2str(Grid_Number)]);
    model.param.set('u_disp', [num2str(u_disp),' [mm]']);
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
    
    % disp(model_filename)
    model.save(model_filename)
    
    model.component('comp1').material.create('mat1', 'Common');
    model.component('comp1').material('mat1').propertyGroup.create('Enu', 'Young''s modulus and Poisson''s ratio');
    
    model.component('comp1').massProp.create('mass1', 'MassProperties');
    model.component('comp1').massProp('mass1').selection.geom('geom1', 3);
    model.component('comp1').massProp('mass1').selection.all;
    model.component('comp1').massProp('mass1').set('densitySource', 'fromPhysics');
    
    model.component('comp1').material('mat1').propertyGroup('Enu').set('E', [num2str(Young),' [GPa]']);
    model.component('comp1').material('mat1').propertyGroup('def').set('density', [num2str(rho),' [kg/m^3]']);
    model.component('comp1').material('mat1').propertyGroup('Enu').set('nu', num2str(nu));
    
    % model.save(model_filename)
    
    model.component('comp1').selection.create('box1', 'Box');
    model.component('comp1').selection('box1').set('entitydim', 2);
    model.component('comp1').selection('box1').label('XP');
    model.component('comp1').selection('box1').set('xmin', 'CellSize');
    model.component('comp1').selection('box1').set('xmax', 'CellSize');
    model.component('comp1').selection('box1').set('ymin', 0);
    model.component('comp1').selection('box1').set('ymax', 'CellSize');
    model.component('comp1').selection('box1').set('zmin', 0);
    model.component('comp1').selection('box1').set('zmax', 'CellSize');
    model.component('comp1').selection('box1').set('condition', 'inside');
    
    model.component('comp1').selection.create('box2', 'Box');
    model.component('comp1').selection('box2').set('entitydim', 2);
    model.component('comp1').selection('box2').label('XN');
    model.component('comp1').selection('box2').set('xmin', 0);
    model.component('comp1').selection('box2').set('xmax', 0);
    model.component('comp1').selection('box2').set('ymin', 0);
    model.component('comp1').selection('box2').set('ymax', 'CellSize');
    model.component('comp1').selection('box2').set('zmin', 0);
    model.component('comp1').selection('box2').set('zmax', 'CellSize');
    model.component('comp1').selection('box2').set('condition', 'inside');
    
    model.component('comp1').selection.create('box3', 'Box');
    model.component('comp1').selection('box3').set('entitydim', 2);
    model.component('comp1').selection('box3').label('YP');
    model.component('comp1').selection('box3').set('xmin', 0);
    model.component('comp1').selection('box3').set('xmax', 'CellSize');
    model.component('comp1').selection('box3').set('ymin', 'CellSize');
    model.component('comp1').selection('box3').set('ymax', 'CellSize');
    model.component('comp1').selection('box3').set('zmin', 0);
    model.component('comp1').selection('box3').set('zmax', 'CellSize');
    model.component('comp1').selection('box3').set('condition', 'inside');
    
    model.component('comp1').selection.create('box4', 'Box');
    model.component('comp1').selection('box4').set('entitydim', 2);
    model.component('comp1').selection('box4').label('YN');
    model.component('comp1').selection('box4').set('xmin', 0);
    model.component('comp1').selection('box4').set('xmax', 'CellSize');
    model.component('comp1').selection('box4').set('ymin', 0);
    model.component('comp1').selection('box4').set('ymax', 0);
    model.component('comp1').selection('box4').set('zmin', 0);
    model.component('comp1').selection('box4').set('zmax', 'CellSize');
    model.component('comp1').selection('box4').set('condition', 'inside');
    
    model.component('comp1').selection.create('box5', 'Box');
    model.component('comp1').selection('box5').set('entitydim', 2);
    model.component('comp1').selection('box5').label('ZP');
    model.component('comp1').selection('box5').set('xmin', 0);
    model.component('comp1').selection('box5').set('xmax', 'CellSize');
    model.component('comp1').selection('box5').set('ymin', 0);
    model.component('comp1').selection('box5').set('ymax', 'CellSize');
    model.component('comp1').selection('box5').set('zmin', 'CellSize');
    model.component('comp1').selection('box5').set('zmax', 'CellSize');
    model.component('comp1').selection('box5').set('condition', 'inside');
    
    model.component('comp1').selection.create('box6', 'Box');
    model.component('comp1').selection('box6').set('entitydim', 2);
    model.component('comp1').selection('box6').label('ZN');
    model.component('comp1').selection('box6').set('xmin', 0);
    model.component('comp1').selection('box6').set('xmax', 'CellSize');
    model.component('comp1').selection('box6').set('ymin', 0);
    model.component('comp1').selection('box6').set('ymax', 'CellSize');
    model.component('comp1').selection('box6').set('zmin', 0);
    model.component('comp1').selection('box6').set('zmax', 0);
    model.component('comp1').selection('box6').set('condition', 'inside');
    
    model.component('comp1').selection.create('uni1', 'Union');
    model.component('comp1').selection('uni1').set('entitydim', 2);
    model.component('comp1').selection('uni1').label('Union X');
    model.component('comp1').selection('uni1').set('input', {'box1' 'box2'});
    
    model.component('comp1').selection.create('uni2', 'Union');
    model.component('comp1').selection('uni2').set('entitydim', 2);
    model.component('comp1').selection('uni2').label('Union Y');
    model.component('comp1').selection('uni2').set('input', {'box3' 'box4'});
    
    model.component('comp1').selection.create('uni3', 'Union');
    model.component('comp1').selection('uni3').set('entitydim', 2);
    model.component('comp1').selection('uni3').label('Union Z');
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


    model.component('comp1').physics('solid').create('disp13', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp13').selection.named('box1');
    model.component('comp1').physics('solid').feature('disp13').set('Direction', [1; 1; 0]);
    model.component('comp1').physics('solid').feature('disp13').set('U0', {'0'; '0'; '0'});
    
    model.component('comp1').physics('solid').create('disp14', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp14').selection.named('box2');
    model.component('comp1').physics('solid').feature('disp14').set('Direction', [1; 1; 0]);
    model.component('comp1').physics('solid').feature('disp14').set('U0', {'0'; '0'; '0'});
    
    model.component('comp1').physics('solid').create('disp15', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp15').selection.named('box3');
    model.component('comp1').physics('solid').feature('disp15').set('Direction', [1; 1; 0]);
    model.component('comp1').physics('solid').feature('disp15').set('U0', {'0'; '0'; '0'});
    
    model.component('comp1').physics('solid').create('disp16', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp16').selection.named('box4');
    model.component('comp1').physics('solid').feature('disp16').set('Direction', [1; 1; 0]);
    model.component('comp1').physics('solid').feature('disp16').set('U0', {'0'; '0'; '0'});
    
    model.component('comp1').physics('solid').create('disp17', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp17').selection.named('box5');
    model.component('comp1').physics('solid').feature('disp17').set('Direction', [1; 1; 1]);
    model.component('comp1').physics('solid').feature('disp17').set('U0', {'0'; '0'; 'u_disp'});
    
    model.component('comp1').physics('solid').create('disp18', 'Displacement2', 2);
    model.component('comp1').physics('solid').feature('disp18').selection.named('box6');
    model.component('comp1').physics('solid').feature('disp18').set('Direction', [1; 1; 1]);
    model.component('comp1').physics('solid').feature('disp18').set('U0', {'0'; '0'; '0'});

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

    model.study.create('std1');
    model.study('std1').create('eig', 'Eigenfrequency');
    model.study('std1').feature('eig').set('neigsactive', true);
    model.study('std1').feature('eig').set('neigs', Number_of_Eigenfrequencies);
    model.study('std1').feature('eig').set('useadvanceddisable', true);
    model.study('std1').feature('eig').set('disabledphysics', {'solid/disp13' 'solid/disp14' 'solid/disp15' 'solid/disp16' 'solid/disp17' 'solid/disp18'});
    model.study('std1').feature('eig').set('useparam', true);
    % model.study('std1').feature('eig').setIndex('pname', 'CellSize', 0);
    % model.study('std1').feature('eig').setIndex('plistarr', '', 0);
    % model.study('std1').feature('eig').setIndex('punit', 'm', 0);
    % model.study('std1').feature('eig').setIndex('pname', 'CellSize', 0);
    % model.study('std1').feature('eig').setIndex('plistarr', '', 0);
    % model.study('std1').feature('eig').setIndex('pname', 'Grid_Number', 0);
    % model.study('std1').feature('eig').setIndex('pname', 'u_disp', 0);
    model.study('std1').feature('eig').setIndex('pname', 'kz', 0);
    model.study('std1').feature('eig').setIndex('plistarr', 'range(0,pi/CellSize/10,pi/CellSize)', 0);
    model.study('std1').feature('eig').setIndex('punit', 'rad/m', 0);

    model.study.create('std2');
    model.study('std2').create('stat', 'Stationary');
    model.study('std2').feature('stat').setSolveFor('/physics/solid', true);
    model.study('std2').feature('stat').set('useadvanceddisable', true);
    model.study('std2').feature('stat').set('disabledphysics', {'solid/pc1' 'solid/pc2' 'solid/pc3'});

    model.save(model_filename)

    model.study('std1').run
    model.study('std2').run

    model.save(model_filename)
    
   
    Frqs = mphglobal(model,'solid.freq', 'dataset', 'dset1');
    kzs = mphglobal(model,'kz', 'dataset', 'dset1');
    frequencies = Frqs (Number_of_Eigenfrequencies:Number_of_Eigenfrequencies:end);
    kz=kzs(Number_of_Eigenfrequencies:Number_of_Eigenfrequencies:end);
    
    nearestFreq2 = 0;
    nearestFreq1 = 0;
    
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
    

    Mass_data=mpheval(model,'mass1.mass', 'dataset', 'dset1');
    MassValue=Mass_data.d1(1);
    Total_Volume=(CellSize/1000)^3;
    DensityStructure=MassValue/Total_Volume;
    Cube_Volume=((CellSize/1000)/(Grid_Number))^3;
    
    Impedance_Dynamic3=Slop*DensityStructure;
    VolumeFraction = (Number_of_Cubes*Cube_Volume)/Total_Volume;
    
    UEnergy3=mphglobal(model,'solid.Ws_tot', 'dataset', 'dset2');
    
    C33=2*UEnergy3/(CellSize/1000)^3/(u_disp/CellSize)^2;
      
    Impedance_Static3 = sqrt(C33*DensityStructure);
        
    Stiffness3 = C33;
   
end