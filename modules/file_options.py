def save_data(data_name, phi, residuals):
    # save the data
    import os
    from dill import dump
    
    print('Saving data...', end='')
    if not os.path.exists('data'):
        os.makedirs('data')

    save_obj = [phi, residuals]
    with open(get_directory() + data_name + '.pkl', 'wb') as file:
        dump(save_obj, file)
    
    print('Done!')

def load_case_individual(case_name):
    # load the case
    
    from dill import load
    from scipy.sparse import load_npz
    import warnings
    
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        with open(get_directory() + case_name + '.pkl', 'rb') as file:
            inputs, indices, rhs, field_functions, ds, sum_nb, TPB_dict, bc_dict = load(file)
    
    J = [None]*3
    for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p!=2:
            continue
        J[p] = load_npz(get_directory() + case_name + f'_sparse_{p}.npz').tolil()

    print('Done!')
    return inputs, indices, J, rhs, field_functions, ds, sum_nb, TPB_dict, bc_dict

def save_case_individual(case_name, 
        inputs, indices, J, rhs,
        field_functions, ds, sum_nb, TPB_dict, bc_dict):
    # save the case
    from dill import dump
    from scipy.sparse import save_npz
    import warnings
    print('Saving case file...', end='')

    save_obj = [
        inputs, indices, rhs,
        field_functions, ds, sum_nb, TPB_dict, bc_dict]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(get_directory() + case_name + '.pkl', 'wb') as case_file:
            dump(save_obj, case_file)
    
    for p in [0,1,2]:
        if inputs['solver_options']['ion_only'] and p!=2:
            continue
        save_npz(get_directory() + case_name + f'_sparse_{p}.npz', J[p].tocsr())

    print('Done!')

def load_case_data_individual(case_name, data_name):
    # load the case
    from dill import load
    inputs, indices, J, rhs, field_functions, ds, sum_nb, TPB_dict, bc_dict = load_case_individual(case_name)
    # load the data
    print('Loading data file...', end='')
    with open(get_directory() + data_name + '.pkl', 'rb') as data_file:
        phi, residuals = load(data_file)
    print('Done!')
    return inputs, indices, J, rhs, field_functions, ds, sum_nb, TPB_dict, bc_dict, phi, residuals

def get_directory():
    from os import getlogin
    
    print('Loading case file...', end='')
    
    username = getlogin()
    if username=='x67637ha' or username=='ASUS': # my own laptop or the university laptop
        directory = 'C:/Users/' + username + '/OneDrive -  The University of Manchester/SOFC/Python case and data/'
    if username=='Hamid': # server computer
        directory = 'D:/Share/Hamid Abbasi/Micromodel/Python case and data/'
    else:
        raise ValueError('Username not recognised')
    return directory 
