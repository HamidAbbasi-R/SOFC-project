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
j0_a = 1e5;     %[A/m2]
j0_c = 1e3;     %[A/m2]
V_op = 1;     %[V]

%% Load\initialize\run\postprocess the COMSOL model
model = mphload('oneD_SOFC.mph');
model.param.set('j0_a', [num2str(j0_a) ' [A/m^2]']);
model.param.set('j0_c', [num2str(j0_c) ' [A/m^2]']);
model.param.set('V_op', [num2str(V_op) ' [V]']);

model.study('std1').run;
PHI_a = model.result.numerical('pev2').getReal();
PHI_c = model.result.numerical('pev1').getReal();
disp(PHI_a)

%% Run python model
Ia_m = py.individual_systems.solve_individual_systems(PHI_a(1), PHI_a(2), PHI_a(3));

