#-----------------------------------------#
# Statistic on density and temperature    #
# input files are dump.xyz and comput.out #
# compute.out using atom-based results    #
# using 2D mapping for equilibrium state  #

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import os

class statistic_2D_equilibrium():

    def __init__(self,dump_interval,time_step,N_frame,N_bin_a,N_bin_b,direction_for_average):

        self.nframes = int(N_frame)
        self.N_bin_a = int(N_bin_a)                 # number of bins in given direction
        self.N_bin_b = int(N_bin_b)
        self.direction_for_average = int(direction_for_average)  # direction for averaging, 0 for bc-plane, 1 for ac-plane, 2 for ab-plane
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
        self.CELL = np.array(line.split('"')[1].split(),dtype=float).reshape((3,3))
        self.CELL_inv = np.linalg.inv(self.CELL)
        self.cell_a = np.linalg.norm(self.CELL[0])
        self.cell_b = np.linalg.norm(self.CELL[1])
        self.cell_c = np.linalg.norm(self.CELL[2])
        #print("Direction index of largest dimension: ", self.direc_index)
        # set bin array in the direction of largest dimension
        if self.direction_for_average == 0:
            print("Mapping to bc plane.")
            self.direct_index_a = 1
            self.direct_index_b = 2
            self.cell_a_cal = self.cell_b
            self.cell_b_cal = self.cell_c
            self.bin_array_a = np.linspace(0, self.cell_b, self.N_bin_a+1)
            self.bin_array_b = np.linspace(0, self.cell_c, self.N_bin_b+1)
        elif self.direction_for_average == 1:
            print("Mapping to ac plane.")
            self.direct_index_a = 0
            self.direct_index_b = 2
            self.cell_a_cal = self.cell_a
            self.cell_b_cal = self.cell_c
            self.bin_array_a = np.linspace(0, self.cell_a, self.N_bin_a+1)
            self.bin_array_b = np.linspace(0, self.cell_c, self.N_bin_b+1)
        elif self.direction_for_average == 2:
            print("Mapping to ab plane.")
            self.direct_index_a = 0
            self.direct_index_b = 1
            self.cell_a_cal = self.cell_a
            self.cell_b_cal = self.cell_b
            self.bin_array_a = np.linspace(0, self.cell_a, self.N_bin_a+1)
            self.bin_array_b = np.linspace(0, self.cell_b, self.N_bin_b+1)
        else:
            raise ValueError("Invalid direction index.")

        tag = []
        for i in range(self.natoms):
            line = f.readline()
            tag.append(line.split()[0])
        # tag is the atomic label of each atom
        self.atomic_tag = np.array(tag)
        self.atomic_type = np.unique(self.atomic_tag)
        self.Natom_at_bin = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b))   # number of atoms in each bin
        self.Temperature = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b))    # temperature in each bin
        self.Enthalpy = np.zeros((len(self.atomic_type),self.N_bin_a,self.N_bin_b))       # enthalpy in each bin H = E + PV
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
        pos_per_frame = np.zeros((self.natoms,2))         # consider only position in concern direction
        potential_per_frame = np.zeros(self.natoms)       # potential energy per atom in each frame
        result_time_type = np.zeros((self.nframes, 4, len(self.atomic_type), self.N_bin_a, self.N_bin_b))  # save results for each frame
        result_time_total = np.zeros((self.nframes, 4, self.N_bin_a, self.N_bin_b))  # save results for each frame
        # result_save[:, :, :, :, 0] = bin_count_atoms
        # result_save[:, :, :, :, 1] = bin_ave_temp
        # result_save[:, :, :, :, 2] = bin_ave_enthalpy
        # result_save[:, :, :, :, 3] = bin_sums_enthalpy

        # result_save[:, :, :, 0] = bin_ave_enthalpy_total
        # result_save[:, :, :, 1] = bin_sums_enthalpy_total

        for t in range(self.nframes):
            line = f.readline()
            line = f.readline()
            for j in range(self.natoms):
                line = f.readline()
                data = line.split()
                # position projected in to the plane of interest
                temp_xyz = np.array(data[1:4], dtype=float)
                temp_abc_reduce = np.dot(self.CELL_inv, temp_xyz)  # convert to fractional coordinates
                temp_a = temp_abc_reduce[self.direct_index_a] * self.cell_a_cal  # convert back to Cartesian coordinates in the direction of interest
                temp_b = temp_abc_reduce[self.direct_index_b] * self.cell_b_cal  # convert back to Cartesian coordinates in the direction of interest

                pos_per_frame[j,:] = [temp_a,temp_b]  # position in the direction of interest
                potential_per_frame[j] = float(data[4])*eV_to_J     # potential J
            Enthalpy_perframe = self.Enthalpy_without_potential[t,:]+potential_per_frame

            # statistic per frame
            bin_count_atoms, bin_ave_temp, bin_ave_enthalpy, bin_sums_enthalpy, bin_ave_enthalpy_total, bin_ave_temperature_total, bin_sums_enthalpy_total, bin_count_all_atoms\
                  = self.calculate_2D_perframe(pos_per_frame, self.Temperature[t,:], Enthalpy_perframe)
            result_time_type[t, 0, :, :, :] = bin_count_atoms
            result_time_type[t, 1, :, :, :] = bin_ave_temp
            result_time_type[t, 2, :, :, :] = bin_ave_enthalpy
            result_time_type[t, 3, :, :, :] = bin_sums_enthalpy
            result_time_total[t, 0, :, :] = bin_ave_enthalpy_total
            result_time_total[t, 1, :, :] = bin_ave_temperature_total
            result_time_total[t, 2, :, :] = bin_sums_enthalpy_total
            result_time_total[t, 3, :, :] = bin_count_all_atoms

        '''
        # time average the results considering the result_time_type[:, :, :, :, 0] > 0
        # count result_time_type[t, :, :, :, 0] > 0
        valid_frames_type = (result_time_type[:, 0, :, :, :] > 0)
        valid_frames_total = (result_time_total[:, 2, :, :] > 0)

        frame_count_type = np.sum(valid_frames_type, axis=0)
        frame_count_total = np.sum(valid_frames_total, axis=0)
        
        # avoid division by zero
        frame_count_type[frame_count_type == 0] = 1
        frame_count_total[frame_count_total == 0] = 1

        # time average the results
        result_time_ave = np.zeros((4, len(self.atomic_type), self.N_bin_a, self.N_bin_b))
        result_time_total_ave = np.zeros((3, self.N_bin_a, self.N_bin_b))

        # process each channel separately
        for i in range(4):
            numerator = np.sum(result_time_type[:, i, :, :, :], axis=0)
            result_time_ave[i, :, :, :] = numerator / frame_count_type

        # process the total results in the same way
        for i in range(3):
            numerator = np.sum(result_time_total[:, i, :, :], axis=0)
            result_time_total_ave[i, :, :] = numerator / frame_count_total
        '''
        result_time_ave = np.mean(result_time_type,axis=0)
        result_time_total_ave = np.mean(result_time_total,axis=0)

        # save the results one by one
        for i in range(len(self.atomic_type)):
            # save the bin count atoms
            output_file = folder+"/bin_count_atoms_%s.txt" % self.atomic_type[i]
            np.savetxt(output_file, result_time_ave[0, i, :, :], fmt='%12.6e')
            output_file = folder+"/bin_ave_temp_%s.txt" % self.atomic_type[i]
            np.savetxt(output_file, result_time_ave[1, i, :, :], fmt='%12.6e')
            output_file = folder+"/bin_ave_enthalpy_%s.txt" % self.atomic_type[i]
            np.savetxt(output_file, result_time_ave[2, i, :, :]/eV_to_J, fmt='%12.6e')
            output_file = folder+"/bin_sums_enthalpy_%s.txt" % self.atomic_type[i]
            np.savetxt(output_file, result_time_ave[3, i, :, :]/eV_to_J, fmt='%12.6e')

        output_file = folder+"/bin_ave_enthalpy_system.txt"
        np.savetxt(output_file, result_time_total_ave[0, :, :]/eV_to_J, fmt='%12.6e')
        output_file = folder+"/bin_ave_temp_system.txt"
        np.savetxt(output_file, result_time_total_ave[1, :, :], fmt='%12.6e')
        output_file = folder+"/bin_sums_enthalpy_system.txt"
        np.savetxt(output_file, result_time_total_ave[2, :, :]/eV_to_J, fmt='%12.6e')
        output_file = folder+"/bin_count_atoms_system.txt"
        np.savetxt(output_file, result_time_total_ave[3, :, :], fmt='%12.6e')

        # save a and b bin edges
        np.savetxt(folder+"/bin_edges_a_b.txt", np.column_stack((self.bin_array_a,self.bin_array_b)), fmt='%12.6e')

        # 0-1 space for the plots
        wspace = 0.25    
        hspace = 0.25    

        # 0-1 margins for the subplots
        left    = 0.08           
        right   = 0.95  
        bottom  = 0.08           
        top     = 0.95      

        bottom2 = 0.12          # bottom margin for the second row of subplots
        top2    = 0.92          # top margin for the second row of subplots

        # plot the results using pcolormesh
        for i in range(len(self.atomic_type)):
            # add one types with four subplots
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_ave[0, i, :, :].T, shading='auto')
            plt.colorbar(label='Number of Atoms')
            plt.xlabel('Position in direction A')
            plt.ylabel('Position in direction B')
            plt.title(f'Number of Atoms for {self.atomic_type[i]}')
            plt.subplot(2, 2, 2)
            plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_ave[1, i, :, :].T, shading='auto')
            plt.colorbar(label='Average Temperature density (K)')
            plt.xlabel('Position in direction A')
            plt.ylabel('Position in direction B')
            plt.title(f'Average Temperature for {self.atomic_type[i]}')
            plt.subplot(2, 2, 3)
            plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_ave[2, i, :, :].T/eV_to_J, shading='auto')
            plt.colorbar(label='Average Enthalpy (eV)')
            plt.xlabel('Position in direction A')
            plt.ylabel('Position in direction B')
            plt.title(f'Average Enthalpy for {self.atomic_type[i]}')
            plt.subplot(2, 2, 4)
            plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_ave[3, i, :, :].T/eV_to_J, shading='auto')
            plt.colorbar(label='Total Enthalpy density (eV)')
            plt.xlabel('Position in direction A')
            plt.ylabel('Position in direction B')
            plt.title(f'Total Enthalpy for {self.atomic_type[i]}')

            plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
            plt.savefig(folder + f'/2D_mapping_{self.atomic_type[i]}.png', dpi=600)
            plt.close()

        # plot the total enthalpy
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_total_ave[1, :, :].T, shading='auto')
        plt.colorbar(label='Temperature spatial density (K)')
        plt.xlabel('Position in direction A')
        plt.ylabel('Position in direction B')
        plt.title('Temperature for System')
        plt.subplot(2, 2, 2)
        plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_total_ave[0, :, :].T/eV_to_J, shading='auto')
        plt.colorbar(label='System enthalpy density (eV)')
        plt.xlabel('Position in direction A')
        plt.ylabel('Position in direction B')
        plt.title('Total Enthalpy for System')
        plt.subplot(2, 2, 3)
        plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_total_ave[2, :, :].T/eV_to_J, shading='auto')
        plt.colorbar(label='Total Enthalpy density (eV)')
        plt.xlabel('Position in direction A')
        plt.ylabel('Position in direction B')
        plt.title('Total Enthalpy for System')
        plt.subplot(2, 2, 4)
        plt.pcolormesh(self.bin_array_a, self.bin_array_b, result_time_total_ave[3, :, :].T, shading='auto')
        plt.colorbar(label='Number of Atoms')
        plt.xlabel('Position in direction A')
        plt.ylabel('Position in direction B')
        plt.title('Number of Atoms for System')
        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plt.savefig(folder + '/2D_mapping_system.png', dpi=600)
        plt.close()

        # plot the results using NonUniformImage
        xc = (self.bin_array_a[:-1] + self.bin_array_a[1:]) / 2
        yc = (self.bin_array_b[:-1] + self.bin_array_b[1:]) / 2
        np.savetxt(folder + "/bin_centers_a_b.txt", np.column_stack((xc, yc)), fmt='%12.6e')
         
        for i in range(len(self.atomic_type)):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            ax = axes[0, 0]  # create an empty plot for NonUniformImage
            img = NonUniformImage(ax, interpolation='bilinear')
            img.set_data(xc, yc, result_time_ave[0, i, :, :].T)
            ax.add_image(img)
            ax.set_xlabel('Position in direction A')
            ax.set_ylabel('Position in direction B')
            ax.set_xlim(self.bin_array_a[0], self.bin_array_a[-1])
            ax.set_ylim(self.bin_array_b[0], self.bin_array_b[-1])
            ax.set_title(f'Number of Atoms for {self.atomic_type[i]}')
            plt.colorbar(img, ax=ax, label='Number of Atoms')
            ax = axes[0, 1]  # create an empty plot for NonUniformImage
            img = NonUniformImage(ax, interpolation='bilinear')
            img.set_data(xc, yc, result_time_ave[1, i, :, :].T)
            ax.add_image(img)
            ax.set_xlabel('Position in direction A')
            ax.set_ylabel('Position in direction B')
            ax.set_xlim(self.bin_array_a[0], self.bin_array_a[-1])
            ax.set_ylim(self.bin_array_b[0], self.bin_array_b[-1])
            ax.set_title(f'Average Temperature for {self.atomic_type[i]}')
            plt.colorbar(img, ax=ax, label='Average Temperature (K)')
            ax = axes[1, 0]  # create an empty plot for NonUniformImage
            img = NonUniformImage(ax, interpolation='bilinear')
            img.set_data(xc, yc, result_time_ave[2, i, :, :].T/eV_to_J)
            ax.add_image(img)
            ax.set_xlabel('Position in direction A')
            ax.set_ylabel('Position in direction B')
            ax.set_xlim(self.bin_array_a[0], self.bin_array_a[-1])
            ax.set_ylim(self.bin_array_b[0], self.bin_array_b[-1])
            ax.set_title(f'Average Enthalpy for {self.atomic_type[i]}')
            plt.colorbar(img, ax=ax, label='Average Enthalpy (eV)')
            ax = axes[1, 1]  # create an empty plot for NonUniformImage
            img = NonUniformImage(ax, interpolation='bilinear')
            img.set_data(xc, yc, result_time_ave[3, i, :, :].T)
            ax.add_image(img)
            ax.set_xlabel('Position in direction A')
            ax.set_ylabel('Position in direction B')
            ax.set_xlim(self.bin_array_a[0], self.bin_array_a[-1])
            ax.set_ylim(self.bin_array_b[0], self.bin_array_b[-1])
            ax.set_title(f'Total Enthalpy for {self.atomic_type[i]}')
            plt.colorbar(img, ax=ax, label='Total Enthalpy (eV)')
            plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
            plt.savefig(folder + f'/2D_mapping_nonuniform_{self.atomic_type[i]}.png', dpi=600)
            plt.close(fig)

        # plot the total enthalpy with zoom in the lattice region #
        low_x = self.bin_array_a[0]
        high_x = min(self.bin_array_a[-1], low_x + 15)
        low_y = self.bin_array_b[0]
        high_y = min(self.bin_array_b[-1], low_y + 15)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes[0, 0]
        temp_density = result_time_total_ave[1, :, :].T
        max_value = np.max(temp_density)
        min_value = np.min(temp_density)
        img = NonUniformImage(ax, interpolation='bilinear',norm=plt.Normalize(vmin=min_value, vmax=max_value))
        img.set_data(xc, yc, temp_density.T)
        ax.add_image(img)
        ax.set_xlabel('Position in direction A')
        ax.set_ylabel('Position in direction B')

        ax.set_xlim(low_x, high_x)
        ax.set_ylim(low_y, high_y)
        ax.set_title(f'Temperature for system')
        plt.colorbar(img, ax=ax, label='Temperature spatial density (K)')

        ax = axes[0, 1]
        max_value = np.max(abs(result_time_total_ave[0, :, :].T/eV_to_J))
        min_value = -max_value
        img = NonUniformImage(ax, interpolation='bilinear',norm=plt.Normalize(vmin=min_value, vmax=max_value))
        img.set_data(xc, yc, result_time_total_ave[0, :, :].T/eV_to_J)
        ax.add_image(img)
        ax.set_xlabel('Position in direction A')
        ax.set_ylabel('Position in direction B')
        ax.set_xlim(low_x, high_x)
        ax.set_ylim(low_y, high_y)
        ax.set_title(f'Ionic Average Enthalpy for System')
        plt.colorbar(img, ax=ax, label='Ionic Average enthalpy density (eV)')

        ax = axes[1, 0]
        max_value = np.max(abs(result_time_total_ave[2, :, :].T/eV_to_J))
        min_value = -max_value
        img = NonUniformImage(ax, interpolation='bilinear',norm=plt.Normalize(vmin=min_value, vmax=max_value))
        img.set_data(xc, yc, result_time_total_ave[2, :, :].T/eV_to_J)
        ax.add_image(img)
        ax.set_xlabel('Position in direction A')
        ax.set_ylabel('Position in direction B')
        ax.set_xlim(low_x, high_x)
        ax.set_ylim(low_y, high_y)
        ax.set_title(f'Enthalpy Landscape for System')
        plt.colorbar(img, ax=ax, label='Total enthalpy density (eV)')

        ax = axes[1, 1]
        max_value = np.max(abs(result_time_total_ave[3, :, :].T))
        min_value = 0
        img = NonUniformImage(ax, interpolation='bilinear',norm=plt.Normalize(vmin=min_value, vmax=max_value))
        img.set_data(xc, yc, result_time_total_ave[3, :, :].T)
        ax.add_image(img)
        ax.set_xlabel('Position in direction A')
        ax.set_ylabel('Position in direction B')
        ax.set_xlim(low_x, high_x)
        ax.set_ylim(low_y, high_y)
        ax.set_title(f'Number density for System')
        plt.colorbar(img, ax=ax, label='Number density (atoms per bin)')

        plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
        plt.savefig(folder + '/2D_mapping_nonuniform_system.png', dpi=600)
        plt.close(fig)
        return

    def calculate_2D_perframe(self, pos_value, temperature_value, enthalpy_value):
        # 初始化数组 [n_types, N_bin_a, N_bin_b]
        bin_count_atoms = np.zeros((len(self.atomic_type), self.N_bin_a, self.N_bin_b))
        bin_sums_temperature = np.zeros_like(bin_count_atoms)
        bin_sums_enthalpy = np.zeros_like(bin_count_atoms)
        bin_ave_temp = np.zeros_like(bin_count_atoms)
        bin_ave_enthalpy = np.zeros_like(bin_count_atoms)

        # 计算总统计量（所有原子类型）
        bin_count_atoms_all, _, _ = np.histogram2d(
            pos_value[:, 0], pos_value[:, 1],
            bins=(self.bin_array_a, self.bin_array_b)
        )
        
        bin_sums_temperature_total, _, _ = np.histogram2d(
            pos_value[:, 0], pos_value[:, 1],
            bins=(self.bin_array_a, self.bin_array_b),
            weights=temperature_value
        )
        
        bin_sums_enthalpy_total, _, _ = np.histogram2d(
            pos_value[:, 0], pos_value[:, 1],
            bins=(self.bin_array_a, self.bin_array_b),
            weights=enthalpy_value
        )
        
        # 按原子类型分别计算
        for k, atom_type in enumerate(self.atomic_type):
            # 获取当前类型的原子索引
            atom_indices = self.tag_index[atom_type]
            #print(f"Processing atomic type: {atom_type}, number of atoms: {len(atom_indices)}")
            # 跳过空类型
            if len(atom_indices) == 0:
                continue
                
            # 获取当前位置、温度和焓值
            pos = pos_value[atom_indices, :]
            temperature = temperature_value[atom_indices]
            enthalpy = enthalpy_value[atom_indices]
            # 计算2D直方图
            bin_count, _, _ = np.histogram2d(
                pos[:, 0], pos[:, 1], 
                bins=(self.bin_array_a, self.bin_array_b)
            )
            
            # 存储原子数量
            bin_count_atoms[k] = bin_count
            bin_sub_ave_temp = np.zeros_like(bin_count)
            bin_sub_ave_enthalpy = np.zeros_like(bin_count)

            # 计算温度加权和
            bin_sums_temp, _, _ = np.histogram2d(
                pos[:, 0], pos[:, 1], 
                bins=(self.bin_array_a, self.bin_array_b), 
                weights=temperature
            )
            bin_sums_temperature[k] = bin_sums_temp
            
            # 计算焓值加权和
            bin_sums_enth, _, _ = np.histogram2d(
                pos[:, 0], pos[:, 1], 
                bins=(self.bin_array_a, self.bin_array_b), 
                weights=enthalpy
            )
            bin_sums_enthalpy[k] = bin_sums_enth
        
            mask = bin_count > 0
            bin_sub_ave_temp[mask] = bin_sums_temp[mask] / bin_count[mask]
            bin_sub_ave_enthalpy[mask] = bin_sums_enth[mask] / bin_count[mask]
            bin_ave_temp[k,:,:] = bin_sub_ave_temp
            bin_ave_enthalpy[k,:,:] = bin_sub_ave_enthalpy

        # 整体统计（所有原子）
        bin_ave_temperature_total = np.zeros_like(bin_count_atoms_all)
        bin_ave_enthalpy_total = np.zeros_like(bin_count_atoms_all)
    
        # 创建掩码，只处理有原子的区域
        total_mask = bin_count_atoms_all > 0
        bin_ave_temperature_total[total_mask] = bin_sums_temperature_total[total_mask] / bin_count_atoms_all[total_mask]
        bin_ave_enthalpy_total[total_mask] = bin_sums_enthalpy_total[total_mask] / bin_count_atoms_all[total_mask]
        
        return (
            bin_count_atoms,          # 每种原子类型的原子数量 [n_types, N_bin_a, N_bin_b]
            bin_ave_temp,             # 每种原子类型的平均温度 [n_types, N_bin_a, N_bin_b]
            bin_ave_enthalpy,         # 每种原子类型的平均焓值 [n_types, N_bin_a, N_bin_b]
            bin_sums_enthalpy,        # 每种原子类型的焓值总和 [n_types, N_bin_a, N_bin_b]
            bin_ave_enthalpy_total,   # 所有原子的平均焓值 [N_bin_a, N_bin_b]
            bin_ave_temperature_total, # 所有原子的平均温度 [N_bin_a, N_bin_b]
            bin_sums_enthalpy_total,   # 所有原子的焓值总和 [N_bin_a, N_bin_b]
            bin_count_atoms_all       # 所有原子的总数 [N_bin_a, N_bin_b]
        )
    
if __name__ == '__main__':

    folder = "md_split"
    sub_list = ["0.6","0.7","0.8","0.9"]
    N_frame = 3000              # number of frames in dump file
    N_bin_a = 80
    N_bin_b = 80
    direction_for_average = 2   # 0 for a, 1 for b, 2 for c
    
    dump_interval = 1000    # frames
    time_step = 0.001       # ps

    stat = statistic_2D_equilibrium(dump_interval,time_step,N_frame,N_bin_a,N_bin_b,direction_for_average)

    for sub in sub_list:
        job_folder = f"{folder}/{sub}"
        stat.read_model(job_folder)
        stat.read_compute_out(job_folder)
        stat.read_dump_xyz(job_folder)

    print("Model and dump files read successfully.")