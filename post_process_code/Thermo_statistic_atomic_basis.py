#-----------------------------#
# get atomic thermoquantities #


import numpy as np
import sys

class statistic_atomic_basis_equilibrium():

    def __init__(self,N_frame):
        self.nframes = int(N_frame)
        return

    def read_model(self,folder):
        # read model.xyz file #
        # atom label  x y z #
        f = open(folder+"/model.xyz","r")
        line = f.readline()
        self.natoms = int(line.split()[0])
        print("Number of atoms in model file: ", self.natoms)
        line = f.readline()
        tag = []
        for i in range(self.natoms):
            line = f.readline()
            tag.append(line.split()[0])
        # tag is the atomic label of each atom
        self.atomic_tag = np.array(tag)
        self.atomic_type = np.unique(self.atomic_tag)

        # save the index of each tag in the atomic_tag in the dictionary
        self.tag_index = {}
        for i, t in enumerate(self.atomic_tag):
            if t not in self.tag_index:
                self.tag_index[t] = []
            self.tag_index[t].append(i)
        #print(self.tag_index[self.atomic_type[1]])
        print("Atomic types: ", self.atomic_type)
        print("Number of elements: ", [len(self.tag_index[t]) for t in self.atomic_type])
        return

    def read_compute_out(self,folder):
        # read compute.out file #
        # Atomic_temp*natoms, viral-x*natoms, viral-y*natoms, viral-z*natoms #
        # H =  K + P + 1/3*( mv^2 + 1/2 * rij * Fij)

        kb = 1.3806E-23     #J/K
        eV_to_J = 1.60218e-19

        f = open(folder+"/compute.out","r")
        self.Temperature = np.zeros((self.nframes,self.natoms))  # Temperature for each atom in each frame
        Virial = np.zeros((self.nframes,self.natoms,3))  # Virial stress for each atom in each frame
        bath_power = np.zeros((self.nframes, 2))  # heat and cold bath power for each frame
        line = f.readline()
        data = line.split()
        if len(data) == self.natoms * 4 + 2:
            print("Using 3-vector virial stress (GPUMD version < 4.0)...")
            for i in range(self.nframes-1):
                line = f.readline()
                data = line.split()
                self.Temperature[i, :] = np.array(data[0:self.natoms], dtype=float)
                Virial[i+1, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
                Virial[i+1, :, 1] = -np.array(data[self.natoms * 2 : self.natoms * 3 ], dtype=float)
                Virial[i+1, :, 2] = -np.array(data[self.natoms * 3 : self.natoms * 4 ], dtype=float)
                bath_power[i+1, :] = np.array(data[self.natoms * 4 : self.natoms * 4 + 2], dtype=float)  # heat and cold bath power
        elif len(data) == self.natoms * 10 + 2:
            print("Using 9-vector virial stress (GPUMD version >= 4.0)...")
            for i in range(self.nframes-1):
                line = f.readline()
                data = line.split()
                self.Temperature[i, :] = np.array(data[0:self.natoms], dtype=float)
                Virial[i+1, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
                Virial[i+1, :, 1] = -np.array(data[self.natoms * 5 : self.natoms * 6 ], dtype=float)
                Virial[i+1, :, 2] = -np.array(data[self.natoms * 9 : self.natoms * 10 ], dtype=float)
                bath_power[i+1, :] = np.array(data[self.natoms * 10 : self.natoms * 10 + 2], dtype=float)
        else:
            raise ValueError("The number of atoms in compute.out does not match the number in model.xyz.")

        Kinetic = 3/2*kb*self.Temperature # J
        self.Enthalpy_without_potential = Kinetic + 1/3*(Kinetic*2 + 1/3*(Virial[:,:,0]+Virial[:,:,1]+Virial[:,:,2])*eV_to_J) # J
        print("Compute.out file read successfully.")

        return
    
    def read_dump_xyz(self,folder):
        # dump.xyz file #
        # atom label, x y z, vx vy vz, fx fy fz, potential

        eV_to_J = 1.60218e-19
        f = open(folder+"/dump.xyz","r")
        potential_per_frame = np.zeros(self.natoms)     # potential energy per atom in each frame
        Enthalpy = np.zeros((self.nframes, self.natoms))  # Enthalpy per frame

        for t in range(self.nframes):
            if t % 100 == 0:
                print("Processing frame %d/%d" % (t, self.nframes))
            line = f.readline()
            line = f.readline()
            for j in range(self.natoms):
                line = f.readline()
                data = line.split()
                #pos_per_frame[j,:] = [float(data[1]),float(data[2]),float(data[3])]  # position in the direction of interest
                potential_per_frame[j] = float(data[4])*eV_to_J     # potential J
            Enthalpy_perframe = self.Enthalpy_without_potential[t,:]+potential_per_frame
            Enthalpy[t,:] = Enthalpy_perframe/eV_to_J

        # reshape Enthalpy to (nframes, natoms)

        for k, atom_type in enumerate(self.atomic_type):
            atom_indices = self.tag_index[atom_type]
            enthalpy_type = Enthalpy[:, atom_indices]  # get enthalpy for this type of atom
        
            # calculate the distribution of enthalpy
            enthalpy_distribution, bins_edges = np.histogram(enthalpy_type, bins=100, density=True)
            bins_out = (bins_edges[:-1]+bins_edges[1:])/2
            np.savetxt(folder+"/%s_enthalpy_distribution.txt" %atom_type, np.column_stack((bins_out, enthalpy_distribution)))

        print("Dump file read successfully.")
        
        return

    
if __name__ == '__main__':

    N_frame = 3000              # number of frames in dump file

    stat = statistic_atomic_basis_equilibrium(N_frame)
    head = "md_batch_random"
    T_list = ["1000"]

    for T in T_list:
        folder = f"{head}/{T}"
        print(f"Processing subfolder: {T}")
        stat.read_model(folder)
        stat.read_compute_out(folder)
        stat.read_dump_xyz(folder)

    print("Model and dump files read successfully.")