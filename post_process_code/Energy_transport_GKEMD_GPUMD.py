import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class Cal_quantity_GK():

    def __init__(self, timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate):

        self.timesteps_md     = timesteps_md
        self.correlation_time = correlation_time
        self.sample_inter     = sample_inter
        self.dump_inter       = dump_inter
        self.mass = np.array(m)*1.66054e-27 # convert to kg
        self.charge = np.array(q)           # charge in e
        self.T = T
        self.N_GROUPS = len(atomic_label)
        self.group_index = int(group_method)
        self.Label = atomic_label
        self.n_atom_group = np.zeros(self.N_GROUPS,dtype=int)   # number of atoms in each group
        self.n_atom_type = np.zeros(len(self.Label),dtype=int)  # number of atoms in each type
        self.replicate = replicate
        self.N_replicate = int(replicate[0]*replicate[1]*replicate[2])
        # assign atom mass and charge based on the type #
        self.type_mass = {}
        self.type_charge = {}
        for t in range(len(self.Label)):
            tag = self.Label[t]
            self.type_mass[tag] = float(self.mass[t])
            self.type_charge[tag] = float(self.charge[t])
        #print("Type mass mapping: ", self.type_mass)
        #print("Type charge mapping: ", self.type_charge)
        return

    def read_data_from_GPUMD(self,file):

        # read compute.out data #
        data = np.loadtxt(file, dtype ='float')
        # checke the data size #
        if np.size(data,axis=1) == self.N_GROUPS*14+2:  # use 3 term virial, xx yy zz
            print("The data size is not from < 4.0 GPUMD")
            virial_terms = 3

        elif np.size(data,axis=1) == self.N_GROUPS*20+2:    
            # use the 9 term virial (GPUMD version > 4.0)
            # we only use xx yy zz terms here while read 9 terms virial
            print("The data size is from >= 4.0 GPUMD")
            virial_terms = 9

        # Assign the data to the variables based on groups #
        self.total_frames = np.size(data,axis=0)
        self.Temperature = np.zeros((self.N_GROUPS,self.total_frames))
        self.Potential = np.zeros((self.N_GROUPS,self.total_frames))
        self.Virial = np.zeros((self.N_GROUPS,3,self.total_frames))     

        self.J_potential = np.zeros((self.N_GROUPS,3,self.total_frames))
        self.J_kenetic = np.zeros((self.N_GROUPS,3,self.total_frames))
        self.Momentum = np.zeros((self.N_GROUPS,3,self.total_frames))

        for i in range(self.N_GROUPS):
            self.Temperature[i,:] = data[:,i].T    #
            self.Potential[i,:] = data[:,self.N_GROUPS+i].T

        if virial_terms == 3:
            for i in range(self.N_GROUPS):
            # xx-g1 xx-g2 xx-g3, yy-g1 yy-g2 yy-g3, zz-g1 zz-g2 zz-g3
                for j in range(3):
                    self.Virial[i,j,:]      = data[:,self.N_GROUPS*2+i+j*self.N_GROUPS] # 3 term virial here
                    self.J_potential[i,j,:] = data[:,self.N_GROUPS*5+i+j*self.N_GROUPS]   # potential
                    self.J_kenetic[i,j,:]   = data[:,self.N_GROUPS*8+i+j*self.N_GROUPS]    # kenetic                                                
                    self.Momentum[i,j,:]    = data[:,self.N_GROUPS*11+i+j*self.N_GROUPS]     # momentum

        elif virial_terms == 9:
            print("Reading 9 term virial data from GPUMD")
            for i in range(self.N_GROUPS):
                # xx-g1 xx-g2 xx-g3, yy-g1 yy-g2 yy-g3, zz-g1 zz-g2 zz-g3
                self.Virial[i,0,:] = data[:,self.N_GROUPS*2+0*self.N_GROUPS+i] # only read 3 term virial here xx
                self.Virial[i,1,:] = data[:,self.N_GROUPS*2+4*self.N_GROUPS+i] # only read 3 term virial here yy
                self.Virial[i,2,:] = data[:,self.N_GROUPS*2+8*self.N_GROUPS+i] # only read 3 term virial here zz

                for j in range(3):
                    self.J_potential[i,j,:] = data[:,self.N_GROUPS*11+i+j*self.N_GROUPS]   # potential
                    self.J_kenetic[i,j,:]   = data[:,self.N_GROUPS*14+i+j*self.N_GROUPS]    # kenetic                                                
                    self.Momentum[i,j,:]    = data[:,self.N_GROUPS*17+i+j*self.N_GROUPS]      # momentum
        return
    
    def read_model(self,file_model,file_volume):
        # read the model file #

        with open(file_model, "r") as f:
            # 1. Read Header
            line = f.readline()
            if not line:
                raise ValueError("File is empty")
            self.N_atoms = int(line.strip())
            f.readline() # Read the comment/lattice line

            # 2. Initialize temporary containers
            # Using dictionaries is faster than list lookups
            group_counts = {} 
            type_counts = {}
            type_group_map = {}
            
            # 3. Single Loop to Read and Process Data
            for _ in range(self.N_atoms):
                line = f.readline()
                parts = line.split()
                
                if not parts: break # Safety check for empty lines

                # Extract data
                atom_type = parts[0]
                # Ensure the line has enough columns before accessing index
                try:
                    group_id = int(parts[self.group_index + 4])
                except IndexError:
                    print(f"Warning: Line format error: {line.strip()}")
                    continue

                # A. Count Groups
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

                # B. Count Types
                type_counts[atom_type] = type_counts.get(atom_type, 0) + 1

                # C. Map Type -> Group
                # We only need to record this the first time we see a new atom type
                if atom_type not in type_group_map:
                    type_group_map[atom_type] = group_id
            f.close()

            n_groups = set(group_counts)
            n_types = set(type_counts)
            if len(n_groups) != len(n_types):
                print(f"n_groups: {len(n_groups)}, n_types: {len(n_types)}")
                sys.exit("Error: Number of groups does not match number of types in model file.")
            if len(n_groups) != self.N_GROUPS:
                print(f"Expected groups: {self.N_GROUPS}, Found groups: {len(n_groups)}")
                sys.exit("Error: Number of groups in model file does not match N_GROUPS.")
            print("Type to group mapping: ", type_group_map)

            # check if self.Label matches the types found
            for lbl in self.Label:
                if lbl not in type_group_map:
                    print(f"Warning: Atom type '{lbl}' not found in model file.")
                    sys.exit(1)
                else:
                    continue
                
            # 4. Store results back into class attributes
            # Store Group Counts
            # Assuming self.n_atom_group is a list indexed by group ID
            for j in range(self.N_GROUPS):
                count = group_counts.get(j, 0)
                self.n_atom_group[j] = count*self.N_replicate
                print(f"Group {j} number: {self.n_atom_group[j]}")

            # Store Type Counts
            # Assuming self.Label is a list of known labels ['C', 'H', etc.]
            for j in range(len(self.Label)):
                lbl = self.Label[j]
                count = type_counts.get(lbl, 0)
                self.n_atom_type[j] = count
                #print(f"Type {lbl} number: {count}")

            # assign mass and charge based on group, use self.type_mass and self.type_charge
            self.group_mass = np.zeros(self.N_GROUPS)
            self.group_charge = np.zeros(self.N_GROUPS)
            for j in range(self.N_GROUPS):
                # find the type corresponding to this group
                for atom_type, group_id in type_group_map.items():
                    if group_id == j:
                        self.group_mass[j] = self.type_mass[atom_type]
                        self.group_charge[j] = self.type_charge[atom_type]
        print("Group mass mapping: ", self.group_mass)
        print("Group charge mapping: ", self.group_charge)
        self.type_to_group = type_group_map
        #sys.exit()

        # read dump.xyz file to get the volume #
        f = open(file_volume,"r")
        while True:
            line = f.readline().strip()
            # find the last line that starts with "Time" #
            if line.startswith("Time"):
                xyz = line.split('Lattice="')[1].split('"')[0]
                abc = np.array(xyz.split(),dtype=float)
                abc = abc.reshape((3,3))
            elif not line:  # if the line is empty, break the loop
                break
            else:
                continue
        self.volume = np.abs(np.linalg.det(abc))*1E-30  # calculate the volume of the unit cell m^3
        print("Volume of the system: %.6E m^3" %self.volume)
        f.close()
        return

    def calculate_enthalpy(self):

        # using equation from https://pubs.aip.org/aip/jap/article/112/5/054310/916898/Equilibrium-molecular-dynamics-determination-of

        # H = 1/N [ K + P + 1/3*( mv^2 + 1/2 * rij * Fij)]
        kb = 1.3806E-23 #J/K
        eV_to_J = 1.60218e-19
        self.Kinetic = 3/2*kb*self.Temperature # J

        self.Enthalpy = np.zeros((self.N_GROUPS,self.total_frames))

        for i in range(self.N_GROUPS):

            self.Potential[i,:] = self.Potential[i,:]/self.n_atom_group[i]*eV_to_J # J
            self.Virial[i,:,:] = -self.Virial[i,:,:]/self.n_atom_group[i]*eV_to_J # J -1/2*rij*Fij in GPUMD
            self.Enthalpy[i,:] = self.Potential[i,:] + self.Kinetic[i,:] + 1/3*(self.Kinetic[i,:]*2 + 1/3*(self.Virial[i,0,:]+self.Virial[i,1,:]+self.Virial[i,2,:])) # J
            
        return
    
    def calculate_heat_flux(self):

        gpuheat_to_J = np.power(1.60218e-19,1.5)*np.power(1.66054e-27,-0.5) # eV^3/2*amu^-1/2 to J*m/s
        gpumomentum_to_J = np.power(1.60218e-19,0.5)*np.power(1.66054e-27,0.5) # eV^1/2*amu^1/2 to kg*m/s

        self.J_total = np.sum((self.J_potential+self.J_kenetic)*gpuheat_to_J,axis=0) # J*m/s

        self.vel = np.zeros((self.N_GROUPS,3,self.total_frames))
        self.J_enthalpy = np.zeros((3,self.total_frames))

        for i in range(self.N_GROUPS):
            self.vel[i,:,:] = self.Momentum[i,:,:]*gpumomentum_to_J/self.group_mass[i]    # m/s
            self.J_enthalpy[0,:] +=  self.vel[i,0,:] * self.Enthalpy[i,:]     # J*m/s
            self.J_enthalpy[1,:] +=  self.vel[i,1,:] * self.Enthalpy[i,:]     # J*m/s
            self.J_enthalpy[2,:] +=  self.vel[i,2,:] * self.Enthalpy[i,:]     # J*m/s

        self.J_conduct = self.J_total - self.J_enthalpy
        # traspose for following calculation #
        self.J_total = self.J_total.T
        self.J_conduct = self.J_conduct.T
        self.J_enthalpy = self.J_enthalpy.T
        return

    def cal_correlation(self,J1,J2):

        self.cal_frames = int(self.total_frames/self.sample_inter)
        
        self.total_time = self.total_frames*self.dump_inter*self.timesteps_md
        self.cor_frames = int(self.correlation_time/self.timesteps_md/\
                              self.sample_inter/self.dump_inter)

        cal_f1 = J1[::self.sample_inter,:]
        cal_f2 = J2[::self.sample_inter,:]

        #print(cal_f_total[0,:])
        average_number = self.cal_frames - self.cor_frames

        correlated_f_total = np.zeros((self.cor_frames,3))  

        for i in range(self.cor_frames):
            correlated_f_total[i,:]=np.average(cal_f1[0:average_number,0:3]*\
                                               cal_f2[0+i:average_number+i,0:3],axis=0)

        correlation_time_list = np.arange(self.cor_frames)*self.timesteps_md*\
                                            self.sample_inter*self.dump_inter
        
        #----------- using reduced unit, kb = 1 -------------#
        scaled_total = correlated_f_total*self.timesteps_md*self.dump_inter*\
                                            self.sample_inter

        return scaled_total, correlation_time_list
    
    def cal_accumulative_acf(self,J1,J2):
        kb = 1.3806E-23 # J/K
        scaled_total, correlation_time_list = self.cal_correlation(J1,J2)
        ccf_total = np.average(scaled_total[:,0:3],axis = 1)/self.volume/self.T/self.T/kb
        converter = 1e-12 # dt unit ps defined here, convert to s
        ccf_total = ccf_total*converter
        
        # ---- Accumulative correlation function ---- #
        sum_ccf = ccf_total*0.0
        sum_ccf[0] = (ccf_total[0]+ccf_total[-1])*0.5 

        for i in range(1,np.size(ccf_total)-1):
            sum_ccf[i] = ccf_total[i] + sum_ccf[i-1]

        sum_ccf[-1] = sum_ccf[-2]
        return scaled_total,  ccf_total,  sum_ccf, correlation_time_list

    def cal_electric_flux(self,cation):

        cation_group = self.type_to_group[cation]
        e_to_C = 1.60218e-19  # e to C

        J_electric = np.zeros((self.N_GROUPS,3,self.total_frames))  # electric flux
        for i in range(self.N_GROUPS):
            J_electric[i,:,:] = self.group_charge[i]* self.vel[i,:,:]
        J_mass_flux = self.group_mass[cation_group]*self.vel[cation_group,:,:]  # mass flux
        self.J_electric = np.sum(J_electric,axis=0)
        self.J_electric = self.J_electric.T*e_to_C # C*m/s
        self.J_mass_flux = J_mass_flux.T
        return

def collect_results_in_folder(folder,timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate,cation):
    # get available subfolders #
    n = len([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
    if n > 0:
        print("Number of cases: %d in folder %s" %(n, folder))
        kappa = np.zeros((n,8))     # store the results for each temperature
        electric = np.zeros((n,5))  # store the electric flux results for each temperature
        mass = np.zeros((n,5))      # store the mass flux results for each temperature
        # n cases every temperature #
        # if n > 1, need to average the results #
        # loop over n with subfolder #
        k = 0
        for sub in os.listdir(folder):
            subfolder = os.path.join(folder,sub)
            kappa_sub, electric_sub, mass_sub = calculation_in_folder(subfolder,timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate,cation)
            kappa[k,:] = kappa_sub
            electric[k,:] = electric_sub
            mass[k,:] = mass_sub
            k += 1
        np.savetxt(os.path.join(folder,'kappa_%d.txt' %T),kappa,header='Total(value, std), Conduction, Soret, Diffusion',fmt='%12.8f')
        np.savetxt(os.path.join(folder,'electric_%d.txt' %T),electric,header='k_electric(value, std), K12, Seebeck_coefficient',fmt='%12.8f')
        np.savetxt(os.path.join(folder,'mass_%d.txt' %T),mass,header='M11(value, std), M12, Velocity',fmt='%12.8f')

        kappa_ave = [np.mean(kappa[:,0]),np.std(kappa[:,0]),np.mean(kappa[:,2]),np.std(kappa[:,2]),\
                        np.mean(kappa[:,4]),np.std(kappa[:,4]),np.mean(kappa[:,6]),np.std(kappa[:,6])]
        electric_ave = [np.mean(electric[:,0]),np.std(electric[:,0]),\
                        np.mean(electric[:,2]),np.std(electric[:,2]),\
                        np.mean(electric[:,4]),np.std(electric[:,4])]
        mass_ave = [np.mean(mass[:,0]),np.std(mass[:,0]),\
                    np.mean(mass[:,2]),np.std(mass[:,2]),\
                    np.mean(mass[:,4]),np.std(mass[:,4])]

        np.savetxt("kappa_%d_average.txt" % T,np.c_[T,kappa_ave],fmt='%d '+'%12.8f '*8,comments='T(K) Total(value, std), Conduction, Soret, Diffusion')
        np.savetxt("electric_%d_average.txt" % T,np.c_[T,electric_ave],fmt='%d '+'%12.8f '*5,comments='T(K) kappa_electric(value, std), K12, Seebeck_coefficient')
        np.savetxt("mass_%d_average.txt" % T,np.c_[T,mass_ave],fmt='%d '+'%12.8f '*5,comments='T(K) M11(value, std), M12, Velocity')
        return kappa_ave, electric_ave, mass_ave
    else:
        print("One case in folder %s, calculating in this folder" % folder)
        kappa, electric, mass = calculation_in_folder(folder,timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate,cation)
        np.savetxt(os.path.join(folder,'kappa_%d.txt' %T),kappa,header='Total(value, std), Conduction, Soret, Diffusion',fmt='%12.8f')
        np.savetxt(os.path.join(folder,'electric_%d.txt' %T),electric,header='k_electric(value, std), K12, Seebeck_coefficient',fmt='%12.8f')
        np.savetxt(os.path.join(folder,'mass_%d.txt' %T),mass,header='M11(value, std), M12, Velocity',fmt='%12.8f')
        return kappa, electric, mass

def calculation_in_folder(thisfolder,timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate,cation):
    kappa = np.zeros((1,8))     # store the results for each case
    electric = np.zeros((1,5))  # store the electric flux results for each case
    mass = np.zeros((1,5))      # store the mass flux results for each case

    model_file = os.path.join(thisfolder,"model.xyz")
    volume_file = os.path.join(thisfolder,"dump.xyz")
    gpumd_file = os.path.join(thisfolder,"compute.out")

    cal = Cal_quantity_GK(timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate)
    cal.read_model(model_file,volume_file)
    cal.read_data_from_GPUMD(gpumd_file)
    cal.calculate_enthalpy()
    cal.calculate_heat_flux()
    cal.cal_electric_flux(cation)

    # energy flux correlation #
    scaled_total1,  ccf_total1,  sum_ccf1, correlation_time_list1 = cal.cal_accumulative_acf(cal.J_total,cal.J_total)
    scaled_total2,  ccf_total2,  sum_ccf2, correlation_time_list2 = cal.cal_accumulative_acf(cal.J_conduct,cal.J_conduct)
    scaled_total3,  ccf_total3,  sum_ccf3, correlation_time_list3 = cal.cal_accumulative_acf(cal.J_conduct,cal.J_enthalpy)
    scaled_total4,  ccf_total4,  sum_ccf4, correlation_time_list4 = cal.cal_accumulative_acf(cal.J_enthalpy,cal.J_enthalpy)

    # electric flux correlation #
    # Transport coefficients from equilibrium molecular dynamics #
    # J. Chem. Phys. 162, 064111 (2025) #

    scaled_total5,  ccf_total5,  sum_ccf5, correlation_time_list5 = cal.cal_accumulative_acf(cal.J_electric,cal.J_electric)     # k_electric eq. (23) unit: S/m
    scaled_total6,  ccf_total6,  sum_ccf6, correlation_time_list6 = cal.cal_accumulative_acf(cal.J_electric,cal.J_conduct)      # K12 eq. (24)  unit A/m

    ccf_total5 *= T
    sum_ccf5 *= T
    ccf_total6 *= T
    sum_ccf6 *= T

    # mass flux correlation #
    scaled_total7,  ccf_total7,  sum_ccf7, correlation_time_list7 = cal.cal_accumulative_acf(cal.J_mass_flux,cal.J_mass_flux)     # M11 eq. (24)  unit kg*s/m^3
    scaled_total8,  ccf_total8,  sum_ccf8, correlation_time_list8 = cal.cal_accumulative_acf(cal.J_mass_flux,cal.J_conduct)       # M12 eq. (24)  unit kg/s/m

    ccf_total7 *= T
    sum_ccf7 *= T
    ccf_total8 *= T
    sum_ccf8 *= T

    np.savetxt(os.path.join(thisfolder,'Correlation_total_%d.txt' %T),np.c_[correlation_time_list1,ccf_total1,sum_ccf1])
    np.savetxt(os.path.join(thisfolder,'Correlation_conduct_%d.txt' %T),np.c_[correlation_time_list2,ccf_total2,sum_ccf2])
    np.savetxt(os.path.join(thisfolder,'Correlation_soret_%d.txt' %T),np.c_[correlation_time_list3,ccf_total3,sum_ccf3])
    np.savetxt(os.path.join(thisfolder,'Correlation_diffusion_%d.txt' %T),np.c_[correlation_time_list4,ccf_total4,sum_ccf4])
    np.savetxt(os.path.join(thisfolder,'Correlation_electric_%d.txt' %T),np.c_[correlation_time_list5,ccf_total5,sum_ccf5])
    np.savetxt(os.path.join(thisfolder,'Correlation_electric_conduct_K12_%d.txt' %T),np.c_[correlation_time_list6,ccf_total6,sum_ccf6])

    #----- set legend -----#
    plt.figure(figsize=(12,6))
    # set figure size #
    set_font = {'family' : 'serif', 'weight' : 'normal', 'size' : 12}
    plt.rc('font', **set_font)
    plt.subplot(1,2,1)
    plt.plot(correlation_time_list1,ccf_total1[:])
    plt.plot(correlation_time_list1,ccf_total2[:])
    plt.plot(correlation_time_list1,ccf_total3[:])
    plt.plot(correlation_time_list1,ccf_total4[:])
    plt.legend(['Total','Conduction','Soret','Diffusion'])
    plt.xlabel('Correlation Time (ps)') 
    plt.ylabel('Coefficient (W/mK)')
    #plt.ylim(-0.1,0.1)

    plt.subplot(1,2,2)
    plt.plot(correlation_time_list1,sum_ccf1[:])
    plt.plot(correlation_time_list1,sum_ccf2[:])
    plt.plot(correlation_time_list1,sum_ccf3[:])
    plt.plot(correlation_time_list1,sum_ccf4[:])
    plt.legend(['Total','Conduction','Soret','Diffusion'])
    plt.xlabel('Correlation Time (ps)')
    plt.ylabel('Acumulative Coefficient (W/mK)')
    #plt.ylim(-0.5,1.0)
    plt.savefig(os.path.join(thisfolder,'Correlation_%d.png' %T))
    plt.close()
    print("Correlation time: %.4f ps"%cal.correlation_time)
    print("Total:      %.4f W/mK" %np.mean(sum_ccf1[-2000:]))
    print("Conduction: %.4f W/mK" %np.mean(sum_ccf2[-2000:]))
    print("Soret:      %.4f W/mK" %np.mean(sum_ccf3[-2000:]))
    print("Diffusion:  %.4f W/mK" %np.mean(sum_ccf4[-2000:]))
    kappa[0,0] = np.mean(sum_ccf1[-2000:])
    kappa[0,1] = np.std(sum_ccf1[-2000:])
    kappa[0,2] = np.mean(sum_ccf2[-2000:])
    kappa[0,3] = np.std(sum_ccf2[-2000:])
    kappa[0,4] = np.mean(sum_ccf3[-2000:])
    kappa[0,5] = np.std(sum_ccf3[-2000:])
    kappa[0,6] = np.mean(sum_ccf4[-2000:])
    kappa[0,7] = np.std(sum_ccf4[-2000:])

    #----- set electric flux -----#
    
    electric[0,0] = np.mean(sum_ccf5[-2000:])
    electric[0,1] = np.std(sum_ccf5[-2000:])
    electric[0,2] = np.mean(sum_ccf6[-2000:])
    electric[0,3] = np.std(sum_ccf6[-2000:])

    # seebeck coefficient, S = K12/K_electric/T  we use the equation S = dV/dT.
    # https://pubs.acs.org/doi/10.1021/acs.jctc.4c00124
    seebeck = electric[0,2]/electric[0,0]/T # unit: V/K
    electric[0,4] = seebeck

    # thermodiffusion velocity M12/M11
    mass[0,0] = np.mean(sum_ccf7[-2000:])   # M11 unit kg*s/m^3
    mass[0,1] = np.std(sum_ccf7[-2000:])    # M11 unit kg*s/m^3
    mass[0,2] = np.mean(sum_ccf8[-2000:])   # M12 unit kg/(m*s)
    mass[0,3] = np.std(sum_ccf8[-2000:])    # M12 unit kg/(m*s)
    mass[0,4] = np.sqrt(abs(mass[0,2])/mass[0,0])/T    # m/s
    # mass[0,4] follow the same sign with M12, if M12 is negative, the velocity is negative, and vice versa.
    mass[0,4] = -mass[0,4] if mass[0,2] < 0 else mass[0,4]  # m/s

    return kappa, electric, mass

def set_chemical_species(atomic_label,q,m,main_folder):
    # chemical species parameters #
    if os.path.exists(os.path.join(main_folder, chemical_input := "chemical_input.txt")):
        # read chemical_input.txt file #
        print("chemical_input.txt file found, reading values.")
        with open(os.path.join(main_folder, chemical_input), "r") as f:
            lines = f.readlines()
            atomic_label = []
            q = []
            m = []
            for line in lines:
                parts = line.split()
                if len(parts) != 3:
                    print(f"Warning: Line format error in chemical_input.txt: {line.strip()}")
                    continue
                atomic_label.append(parts[0])
                q.append(float(parts[1]))
                m.append(float(parts[2]))
        f.close()
    else:
        print("chemical_input.txt file not found, using following values.")
        # save chemical_input.txt file #
        with open(os.path.join(main_folder, "chemical_input.txt"), "w") as f:
            for i in range(len(atomic_label)):
                f.write(f"{atomic_label[i]} {q[i]} {m[i]}\n")
        f.close()
    return atomic_label, q, m

if __name__ == '__main__':
    
    # md parameters #
    timesteps_md = 0.001          # ps
    correlation_time = 20         # ps
    sample_inter = 1
    dump_inter = 5                # dump every 5 steps

    # system parameters #
    group_method = 0                # group method index in model.xyz
    replicate = [1,1,1]             # repetitions in x,y,z direction, affect the number of atoms

    cation = 'Ag'                   # index of the cation in atomic_label

    atomic_label = ['Ag','Te']            # atomic label
    q = [1, -2]                           # charge of the atoms in e
    m = [107.87, 127.60]                  # mass of the atoms in amu

    main_folder = "./"  # main folder containing subfolders for different temperatures
    job_list = ["5","6","7","9","10","11"]  # list of subfolders for different temperatures

    atomic_label, q, m = set_chemical_species(atomic_label,q,m,main_folder)
 
    for job in job_list:
        folder = os.path.join(main_folder, job)
        kappa = []
        electric = []
        mass = []
        temperature = []

        for n in range(1, 6, 1):
            T = 500
            # get available subfolders #
            sub_folder = os.path.join(folder, "%d" % n)
            if os.path.exists(sub_folder):
                kappa_T, electric_T, mass_T = collect_results_in_folder(sub_folder,timesteps_md,correlation_time,sample_inter,dump_inter,m,q,T,atomic_label,group_method,replicate,cation)
                # kappa_T shape (1,8)
                # electric_T shape (1,5)
                # mass_T shape (1,5)
                kappa.append(kappa_T)
                electric.append(electric_T)
                mass.append(mass_T)
                temperature.append(T)
            else:
                print("Folder %s does not exist." %sub_folder)
                continue

        # --- Post-Processing and Saving ---

        # Convert lists to numpy arrays
        temperature = np.array(temperature)
        
        # Reshape data arrays (removes the extra dimension if present)
        # kappa becomes (N_samples, 8)
        kappa = np.array(kappa).reshape(-1, 8)
        electric = np.array(electric).reshape(-1, 5)
        mass = np.array(mass).reshape(-1, 5)

        # 1. Save Kappa (Temperature + 8 columns)
        # Stack T and Data side-by-side
        data_kappa = np.column_stack((temperature, kappa))
        header_kappa = 'T(K) Total(value, std) Conduction Soret Diffusion (Total 9 cols)'
        # Create format: 1 integer, then 8 floats
        fmt_kappa = ['%d'] + ['%12.8f'] * 8 
        np.savetxt(os.path.join(folder, "kappa_all.txt"), data_kappa, fmt=fmt_kappa, header=header_kappa)

        # 2. Save Electric (Temperature + 5 columns)
        data_electric = np.column_stack((temperature, electric))
        header_electric = 'T(K) k_electric(value, std) K12 Seebeck_coefficient'
        fmt_electric = ['%d'] + ['%12.8f'] * 5
        np.savetxt(os.path.join(folder, "electric_all.txt"), data_electric, fmt=fmt_electric, header=header_electric)

        # 3. Save Mass (Temperature + 5 columns)
        data_mass = np.column_stack((temperature, mass))
        header_mass = 'T(K) M11(value, std) M12 Velocity'
        fmt_mass = ['%d'] + ['%12.8f'] * 5
        np.savetxt(os.path.join(folder, "mass_all.txt"), data_mass, fmt=fmt_mass, header=header_mass)

        print("All calculations are done.")