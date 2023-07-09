import json

id_start = 111
n_cases = 4

params = {
    # "pH2_inlet": [0.95] * n_cases,
    # "average_diameter": [0.5e-6] * (n_cases//2) + [1e-6] * (n_cases//2),
    # "Vel_b": [0.001, 0.01, 0.1, 0.2, 0.5] * (n_cases//5),
    # "length_X": [10e-6]*5 + [20e-6]*5 + [40e-6]*5 + [50e-6]*5 + [100e-6]*5 + [10e-6]*5 + [20e-6]*5 + [30e-6]*5 + [40e-6]*5 + [100e-6]*5,
    "scale_factor": [1,2,4,6],
    # "length_YZ": [10e-6] * n_cases,
    # "lattice_geometry": [False] * n_cases,
    # "vf_pores": [0.44] * n_cases,
    # "vf_Ni": [0.28] * n_cases,
    # "vf_YSZ": [0.28] * n_cases,
}

f = open('input files/inputs_template.json')
inputs = json.load(f)

for i in range(n_cases):
    id_num = id_start + i
    id_num = str(id_num).zfill(3)
    id_str = "input files/inputs_" + id_num + ".json"

    # inputs['boundary_conditions']['pH2_inlet'] = params['pH2_inlet'][i]
    # inputs['boundary_conditions']['pH2_b'] = params['pH2_inlet'][i]
    # inputs['boundary_conditions']['Vel_b'] = params['Vel_b'][i]
    inputs['microstructure']['scale_factor'] = params['scale_factor'][i]
    # inputs['microstructure']['lattice_geometry']['flag'] = params['lattice_geometry'][i]
    # inputs['microstructure']['length']['X'] = params['length_X'][i]
    # inputs['microstructure']['length']['Y'] = params['length_YZ'][i]
    # inputs['microstructure']['length']['Z'] = params['length_YZ'][i]
    inputs['file_options']['id'] = id_num

    json_object = json.dumps(inputs, indent = 4)
    with open(id_str, "w") as outfile:
        outfile.write(json_object)
