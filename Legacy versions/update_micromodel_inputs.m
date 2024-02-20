function update_micromodel_inputs(pH2_in, pH2_b, Vel_b, Vio_b, T)
jsontext = fileread('inputs.json');
jsonData = jsondecode(jsontext);

jsonData.boundary_conditions.pH2_inlet = pH2_in;
jsonData.boundary_conditions.pH2_b = pH2_b;
jsonData.boundary_conditions.Vel_b = Vel_b;
jsonData.boundary_conditions.Vio_b = Vio_b;
jsonData.operating_conditions.T = T;

jsonText2 = jsonencode(jsonData);

fid = fopen('inputs.json', 'w');
fprintf(fid, '%s', jsonText2);
fclose(fid);