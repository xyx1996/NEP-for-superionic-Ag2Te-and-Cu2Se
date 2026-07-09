#-----------------------------------------#
# Statistic on density and temperature    #
# input files are dump.xyz and comput.out #
# compute.out using atom-based results    #
# using 3D mapping for equilibrium state  #

import numpy as np
import sys
class statistic_3D_equilibrium():

    def __init__(self,dump_interval,time_step,N_frame,N_bin_a,N_bin_b,N_bin_c,):

        self.nframes = int(N_frame)
        self.N_bin_a = int(N_bin_a)                 # number of bins in given direction
        self.N_bin_b = int(N_bin_b)                 # number of bins in given direction
        self.N_bin_c = int(N_bin_c)                 # number of bins in given direction
        self.dump_interval = int(dump_interval)     # dump interval in frames
        self.time_step = float(time_step)           # time step in ps
        return

    def read_model(self,folder):
        # read model.xyz file #
        # atom label  x y z #
        f = open(folder+"/model.xyz","r")
        line = f.readline()
        self.natoms = int(line.split()[0])
        print("Number of atoms in model file: ", self.natoms)
        line = f.readline()
        CELL = np.array(line.split('"')[1].split(),dtype=float).reshape((3,3))
        #print("Direction index of largest dimension: ", self.direc_index)
        # set bin array in the direction of largest dimension


        self.bin_array_a = np.linspace(CELL[:,1].min(), CELL[:,1].max(), self.N_bin_a+1)
        self.bin_array_b = np.linspace(CELL[:,2].min(), CELL[:,2].max(), self.N_bin_b+1)
        self.bin_array_c = np.linspace(CELL[:,0].min(), CELL[:,0].max(), self.N_bin_c+1)

        self.bins =  (self.bin_array_a, self.bin_array_b, self.bin_array_c)  # bins for histogram

        tag = []
        for i in range(self.natoms):
            line = f.readline()
            tag.append(line.split()[0])
        # tag is the atomic label of each atom
        self.atomic_tag = np.array(tag)
        self.atomic_type = np.unique(self.atomic_tag)
        self.Natom_at_bin = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b,self.N_bin_c))   # number of atoms in each bin
        #self.Temperature = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b,self.N_bin_c))    # temperature in each bin
        self.Enthalpy = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b,self.N_bin_c))       # enthalpy in each bin H = E + PV
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

        if len(data) == self.natoms * 4 + 2: # 2 more columns for heat and cold bath power
            print("Using 3-vector virial stress (GPUMD version < 4.0)...")
            self.Temperature[i, :] = np.array(data[0:self.natoms], dtype=float)
            # xx, yy, zz
            Virial[0, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
            Virial[0, :, 1] = -np.array(data[self.natoms * 2 : self.natoms * 3 ], dtype=float)
            Virial[0, :, 2] = -np.array(data[self.natoms * 3 : self.natoms * 4 ], dtype=float)
            bath_power[0, :] = np.array(data[self.natoms * 4 : self.natoms * 4 + 2], dtype=float)  # heat and cold bath power
            for i in range(self.nframes-1):
                line = f.readline()
                data = line.split()
                self.Temperature[i+1, :] = np.array(data[0:self.natoms], dtype=float)
                Virial[i+1, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
                Virial[i+1, :, 1] = -np.array(data[self.natoms * 2 : self.natoms * 3 ], dtype=float)
                Virial[i+1, :, 2] = -np.array(data[self.natoms * 3 : self.natoms * 4 ], dtype=float)
                bath_power[i+1, :] = np.array(data[self.natoms * 4 : self.natoms * 4 + 2], dtype=float)  # heat and cold bath power

        elif len(data) == self.natoms * 10 + 2:
            print("Using 9-vector virial stress (GPUMD version >= 4.0)...")
            self.Temperature[0, :] = np.array(data[0:self.natoms], dtype=float)
            # xx, xy, xz, yx, yy, yz, zx, zy, zz
            Virial[0, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
            Virial[0, :, 1] = -np.array(data[self.natoms * 5 : self.natoms * 6 ], dtype=float)
            Virial[0, :, 2] = -np.array(data[self.natoms * 9 : self.natoms * 10 ], dtype=float)
            bath_power[0, :] = np.array(data[self.natoms * 10 : self.natoms * 10 + 2], dtype=float)
            for i in range(self.nframes-1):
                line = f.readline()
                data = line.split()
                self.Temperature[i+1, :] = np.array(data[0:self.natoms], dtype=float)
                Virial[i+1, :, 0] = -np.array(data[self.natoms * 1 : self.natoms * 2 ], dtype=float)
                Virial[i+1, :, 1] = -np.array(data[self.natoms * 5 : self.natoms * 6 ], dtype=float)
                Virial[i+1, :, 2] = -np.array(data[self.natoms * 9 : self.natoms * 10 ], dtype=float)
                bath_power[i+1, :] = np.array(data[self.natoms * 10 : self.natoms * 10 + 2], dtype=float)

        Kinetic = 3/2*kb*self.Temperature # J
        self.Enthalpy_without_potential = Kinetic + 1/3*(Kinetic*2 + 1/3*(Virial[:,:,0]+Virial[:,:,1]+Virial[:,:,2])*eV_to_J) # J
        print("Compute.out file read successfully.")
        
        bath_power = bath_power * eV_to_J
        time_list = np.arange(self.nframes) * self.dump_interval * self.time_step  # time in ps
        np.savetxt(folder+"/bath_power.txt", np.column_stack((time_list,bath_power)), fmt='%12.6e', 
                   header='Time (ps), Heat bath power (W), Cold bath power (W)', comments='# ')
        return
    
    def read_dump_xyz(self,folder):
        # dump.xyz file #
        # atom label, x y z, vx vy vz, fx fy fz, potential

        eV_to_J = 1.60218e-19
        f = open(folder+"/dump.xyz","r")
        pos_per_frame = np.zeros((self.natoms,3))         # consider only position in concern direction
        potential_per_frame = np.zeros(self.natoms)     # potential energy per atom in each frame
        result_time_type = np.zeros((len(self.atomic_type), self.N_bin_a, self.N_bin_b, self.N_bin_c, 4))  # save results for each frame
        result_time_total = np.zeros((self.N_bin_a, self.N_bin_b, self.N_bin_c, 3))  # save results for each frame
        # result_save[:, :, :, :, :, 0] = bin_count_atoms
        # result_save[:, :, :, :, :, 1] = bin_ave_temp
        # result_save[:, :, :, :, :, 2] = bin_ave_enthalpy
        # result_save[:, :, :, :, :, 3] = bin_sums_enthalpy

        # result_save[:, :, :, :, 0] = bin_ave_enthalpy_total
        # result_save[:, :, :, :, 1] = bin_sums_enthalpy_total

        for t in range(self.nframes):
            if t % 100 == 0:
                print("Processing frame %d/%d" % (t, self.nframes))
            line = f.readline()
            line = f.readline()
            for j in range(self.natoms):
                line = f.readline()
                data = line.split()
                pos_per_frame[j,:] = [float(data[1]),float(data[2]),float(data[3])]  # position in the direction of interest
                potential_per_frame[j] = float(data[4])*eV_to_J     # potential J
            Enthalpy_perframe = self.Enthalpy_without_potential[t,:]+potential_per_frame

            # statistic per frame
            bin_count_atoms, bin_ave_temp, bin_ave_enthalpy, bin_sums_enthalpy, bin_ave_enthalpy_total, \
                bin_sums_enthalpy_total, bin_sums_temperature_total = self.calculate_3D_perframe(pos_per_frame, self.Temperature[t,:], Enthalpy_perframe)
            result_time_type[:, :, :, :, 0] += bin_count_atoms
            result_time_type[:, :, :, :, 1] += bin_ave_temp
            result_time_type[:, :, :, :, 2] += bin_ave_enthalpy
            result_time_type[:, :, :, :, 3] += bin_sums_enthalpy
            result_time_total[:, :, :, 0] += bin_ave_enthalpy_total
            result_time_total[:, :, :, 1] += bin_sums_enthalpy_total
            result_time_total[:, :, :, 2] += bin_sums_temperature_total

        # time average the results
        result_time_type = result_time_type / self.nframes
        #result_time_std = np.std(result_time_type, axis=0)
        result_time_total = result_time_total / self.nframes
        #result_time_total_std = np.std(result_time_total, axis=0)

        # add labels to the result arrays
        for k in range(len(self.atomic_type)):
            np.savez(folder+"/result_time_ave_%s.npz" % self.atomic_type[k], 
                     count=result_time_type[k, :, :, :, 0], 
                     local_temperature=result_time_type[k, :, :, :, 1], 
                     local_atomic_enthalpy=result_time_type[k, :, :, :, 2], 
                     local_enthalpy=result_time_type[k, :, :, :, 3])
        np.savez(folder+"/result_time_total_ave.npz", 
                  local_atomic_enthalpy=result_time_total[:, :, :, 0],
                  local_enthalpy=result_time_total[:, :, :, 1],
                  local_temperature=result_time_total[:, :, :, 2])
        np.save(folder+"/bin_edges.npy", self.bins)
        print("Dump file read successfully.")
        
        return

    def calculate_3D_perframe(self, pos_value, temperature_value, enthalpy_value):
        # Preallocate arrays for histograms
        shape = (len(self.atomic_type), self.N_bin_a, self.N_bin_b, self.N_bin_c)
        bin_count_atoms = np.zeros(shape)
        bin_sums_temperature = np.zeros(shape)
        bin_sums_enthalpy = np.zeros(shape)
        
        # Compute histograms in a single pass per atom type
        for k, atom_type in enumerate(self.atomic_type):
            atom_indices = self.tag_index[atom_type]
            pos = pos_value[atom_indices]
            
            # Compute histograms in one go
            bin_count_atoms[k,:,:,:], _ = np.histogramdd(
                pos, bins=self.bins
            )
            bin_sums_temperature[k,:,:,:], _ = np.histogramdd(
                pos, bins=self.bins, 
                weights=temperature_value[atom_indices]
            )
            bin_sums_enthalpy[k,:,:,:], _ = np.histogramdd(
                pos, bins=self.bins, 
                weights=enthalpy_value[atom_indices]
            )
            #print(np.sum(bin_count_atoms[k,:,:,:]), np.sum(bin_sums_temperature[k,:,:,:])/np.sum(bin_count_atoms[k,:,:,:]))

        # Compute averages using safe division
        bin_ave_temp = np.zeros_like(bin_sums_temperature)
        bin_ave_enthalpy = np.zeros_like(bin_sums_enthalpy)
        
        mask = bin_count_atoms > 0
        bin_ave_temp[mask] = bin_sums_temperature[mask] / bin_count_atoms[mask]
        bin_ave_enthalpy[mask] = bin_sums_enthalpy[mask] / bin_count_atoms[mask]
        #print(np.sum(bin_ave_temp)/np.sum(bin_count_atoms))
        #print(np.sum(bin_ave_temp[0,:,:,:])/np.sum(bin_count_atoms[0,:,:,:]))
        #print(np.sum(bin_ave_temp[1,:,:,:])/np.sum(bin_count_atoms[1,:,:,:]))
        #sys.exit()
        # Compute total enthalpy metrics
        bin_ave_temperature_total = np.sum(bin_sums_temperature, axis=0)
        bin_sums_enthalpy_total = np.sum(bin_sums_enthalpy, axis=0)
        total_atom_count = np.sum(bin_count_atoms, axis=0)
        
        bin_ave_enthalpy_total = np.zeros_like(bin_sums_enthalpy_total)
        total_mask = total_atom_count > 0
        bin_ave_enthalpy_total[total_mask] = (
            bin_sums_enthalpy_total[total_mask] / total_atom_count[total_mask]
        )
        bin_ave_temperature_total[total_mask] = (
            bin_ave_temperature_total[total_mask] / total_atom_count[total_mask]
        )

        #print(np.sum(bin_ave_temperature_total)/np.sum(bin_count_atoms))
        #print(np.max(bin_ave_temperature_total),np.min(bin_ave_temperature_total))
        #sys.exit()

        return (
            bin_count_atoms,
            bin_ave_temp,
            bin_ave_enthalpy,
            bin_sums_enthalpy,
            bin_ave_enthalpy_total,
            bin_sums_enthalpy_total,
            bin_ave_temperature_total
        )
    
if __name__ == '__main__':


    folder = "NVT/md_batch_restart"
    sub_list = ["200","400","600","1000"]

    N_frame = 3000              # number of frames in dump file
    N_bin_a = 40
    N_bin_b = 40
    N_bin_c = 40                # number of bins in each direction

    dump_interval = 1000    # frames
    time_step = 0.001       # ps

    stat = statistic_3D_equilibrium(dump_interval,time_step,N_frame,N_bin_a,N_bin_b,N_bin_c)

    for sub in sub_list:
        job_folder = f"{folder}/{sub}"
        stat.read_model(job_folder)
        stat.read_compute_out(job_folder)
        stat.read_dump_xyz(job_folder)
    print("Model and dump files read successfully.")