import json

id_start = 56
params = {
    # "pH2_inlet": [0.95] * 5,
    # "average_diameter": [0.5e-6] * 5,
    "Vel_b": [0.001, 0.01, 0.1, 0.2, 0.5],
    # "length_X": [30e-6] * 5,
    # "length_YZ": [1e-6] * 5,
    # "lattice_geometry": [True] * 5,
    # "vf_pores": [0.3] * 5,
    # "vf_Ni": [0.35] * 5,
    # "vf_YSZ": [0.3] * 5,
}

n_cases = len(list(params.values())[0])

f = open('input files/inputs_template.json')
inputs = json.load(f)

for i in range(n_cases):
    id_num = id_start + i
    id_num = str(id_num).zfill(3)
    id_str = "input files/inputs_" + id_num + ".json"

    # inputs['boundary_conditions']['pH2_inlet'] = params['pH2_inlet'][i]
    # inputs['boundary_conditions']['pH2_b'] = params['pH2_inlet'][i]
    inputs['boundary_conditions']['Vel_b'] = params['Vel_b'][i]
    # inputs['microstructure']['average_diameter'] = params['average_diameter'][i]
    # inputs['microstructure']['length']['X'] = params['length_X'][i]
    # inputs['microstructure']['length']['Y'] = params['length_YZ'][i]
    # inputs['microstructure']['length']['Z'] = params['length_YZ'][i]
    inputs['file_options']['id'] = id_num

    json_object = json.dumps(inputs, indent = 4)
    with open(id_str, "w") as outfile:
        outfile.write(json_object)
