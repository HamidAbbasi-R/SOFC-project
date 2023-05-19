import json

n_cases = 4
id_start = 17
params = {
    "pH2_inlet": [0.5]*4,
    "average_diameter": [1e-6]*4,
    "Vel_b": [0.001, 0.01, 0.1, 0.2],
    "length_X": [50e-6]*4,
}

f = open('input files/inputs.json')
inputs = json.load(f)

for i in range(n_cases):
    id_num = id_start + i
    id_num = str(id_num).zfill(2)
    id_str = "input files/inputs_" + id_num + ".json"

    inputs['boundary_conditions']['pH2_inlet'] = params['pH2_inlet'][i]
    inputs['boundary_conditions']['pH2_b'] = params['pH2_inlet'][i]
    inputs['boundary_conditions']['Vel_b'] = params['Vel_b'][i]
    inputs['microstructure']['average_diameter'] = params['average_diameter'][i]
    inputs['microstructure']['length']['X'] = params['length_X'][i]
    inputs['file_options']['id'] = id_num

    json_object = json.dumps(inputs, indent = 4)
    with open(id_str, "w") as outfile:
        outfile.write(json_object)
