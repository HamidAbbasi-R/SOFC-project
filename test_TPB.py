from modules.topology import create_ideal_microstructure_spheres as cim

analytical = []
numerical = []

cim((100,
     100,
     100), 7)
for r in range(5,6):
    an,nu = cim(r)
    analytical.append(an)
    numerical.append(nu)
    # write results to a text file
    with open('results.txt', 'a') as f:
        f.write('r = %d \t analytical = %d \t numerical = %d \n' % (r, an, nu))

