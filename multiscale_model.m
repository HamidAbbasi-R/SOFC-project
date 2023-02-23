%% Path management
clc,clear
clear classes
a = pyenv;
b = a.Home + "\Library\bin\libiomp5md.dll";
if isfile(b)
    disp('libiomp5md.dll is found in current python environment!')
else
    disp('libiomp5md.dll is NOT found in current python environment!')
end
username = getenv('username');
COMSOL_version = '61';
COMSOL_software_path = ['C:\Program Files\COMSOL\COMSOL',COMSOL_version,'\Multiphysics\mli'];
COMSOL_files_path = ['C:\Users\',username,'\OneDrive - The University of Manchester\SOFC\COMSOL'];
% Python_path = ['C:\Users\',username,'\OneDrive - The University of Manchester\SOFC\Python'];
addpath(COMSOL_software_path);
addpath(COMSOL_files_path);
% addpath(Python_path);
mod = py.importlib.import_module('individual_systems');
py.importlib.reload(mod);

%% Constant parameters
% anode side area-specific exchange current density (initial guess)
j0_a = 1e5;     %[A/m2]

% cathode side area-specific exchange current density 
j0_c = 1e3;     %[A/m2]

% operating voltage
V_op = 1;     %[V]
tol = 1e-2;

%% Load\initialize\run\postprocess the COMSOL model
model = mphload('oneD_SOFC.mph');
model.param.set('j0_a', [num2str(j0_a) ' [A/m^2]']);
model.param.set('j0_c', [num2str(j0_c) ' [A/m^2]']);
model.param.set('V_op', [num2str(V_op) ' [V]']);

model.study('std1').run;

% get results from MACRO model
% PHI_a is [hydrogen concentration, electron potential, ion potential, current density] at the anode/electrolyte interface
% PHI_c is [oxygen concentration, electron potential, ion potential, current density] at the cathode/electrolyte interface
PHI_a = model.result.numerical('pev2').getReal();
Ia_M = PHI_a(4);

%% Run python model
while true
    % update the inputs of the micromodel
    update_micromodel_inputs(PHI_a);

    % run the micromodel
    Ia_m = py.individual_systems.solve_individual_systems();

    % update the inputs of the MACRO model
    model.param.set('j_a', [num2str(j0_a) ' [A/m^2]']);

    % run the MACRO model
    model.study('std1').run;

    % get results from MACRO model
    PHI_a = model.result.numerical('pev2').getReal();
    Ia_M = PHI_a(4);

    error = abs(Ia_M - Ia_m)/Ia_M;
    if error < tol
        break
    end
end