%% Path management
clc,clear
clear classes
a = pyenv;
b = a.Home + "\Library\bin\libiomp5md.dll";
if isfile(b)
    disp('libiomp5md.dll is found in the current python environment!')
else
    disp('libiomp5md.dll is NOT found in the current python environment!')
end
username = getenv('username');
COMSOL_version = '61';
COMSOL_software_path = ['C:\Program Files\COMSOL\COMSOL',COMSOL_version,'\Multiphysics\mli'];
COMSOL_files_path = ['C:\Users\',username,'\OneDrive - The University of Manchester\SOFC\COMSOL model'];
% Python_path = ['C:\Users\',username,'\OneDrive - The University of Manchester\SOFC\Python'];
addpath(COMSOL_software_path);
addpath(COMSOL_files_path);
% addpath(Python_path);
mod = py.importlib.import_module('individual_systems');
py.importlib.reload(mod);

%% Constant parameters
% anode side area-specific exchange current density (initial guess)
j0_a_0 = 1e5;     %[A/m2]

% cathode side area-specific exchange current density 
j0_c = 1e3;     %[A/m2]

% operating voltage
V_op = 1;     %[V]
tol = 1e-2;

%% Load\initialize\run\postprocess the COMSOL model
model = mphload('oneD_SOFC.mph');
model.param.set('j0_a', [num2str(j0_a_0) ' [A/m^2]']);
model.param.set('j0_c', [num2str(j0_c) ' [A/m^2]']);
model.param.set('V_op', [num2str(V_op) ' [V]']);

% get results from MACRO model
% we need to get the inlet hydrogen partial pressure (pH2_inlet) from the MACRO model as well
% PHI_a is [hydrogen concentration, electron potential, ion potential, current density] at the anode/electrolyte interface
% PHI_c is [oxygen concentration, electron potential, ion potential, current density] at the cathode/electrolyte interface


%% Run python model
i = 1;
j0_a(i) = j0_a_0;
while true
    % run the MACRO model
    model.study('std1').run;

    % get the results of MACRO model
    PHI_a = model.result.numerical('pev1').getReal();
    xH2_in = model.result.numerical('pev3').getReal();
    T = mphevaluate(model, 'T');        % [K]
    pH2_in = xH2_in;        % partial pressure in atm is equal to volume fraction
    pH2_b = PHI_a(1);       % [atm]
    Vel_b = PHI_a(2);       % [V]
    Vio_b = PHI_a(3);       % [V]
    Ia_M = PHI_a(4);        % [A/m2]

    % update the inputs of the micromodel
    update_micromodel_inputs(pH2_in, pH2_b, Vel_b, Vio_b, T);

    % run the micromodel
    Ia_m = py.individual_systems.solve_individual_systems();   % [A/m2]

    % measure the error and check the convergence
    error = abs(Ia_M - Ia_m)/Ia_m;
    if error < tol
        break
    end

    % change MACRO model input
    f = Ia_m / Ia_M;
    j0_a(i+1) = j0_a(i) * f;
    model.param.set('j0_a', [num2str(j0_a(i+1)) ' [A/m^2]']);
    i=i+1;
end