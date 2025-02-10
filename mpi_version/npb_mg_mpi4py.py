import numpy as np
from mpi4py import MPI
from parser import get_parser
import math

''' 
    Simplification/Assumption:
        1. Nx = Ny = Nz
        
'''
# global variables here
comm = MPI.COMM_WORLD
Nx, Ny, Nz = 0, 0, 0
single_process_z_range = 1
process_activate_flag = True
max_num_process = 0
current_grid_level = 0
max_grid_level = 0
local_approximate_sol = None
local_z = None
local_r_list = []

def setup():
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z, local_r_list

    setup_debug = False

    curr_rank = comm.Get_rank()
    args = get_parser().parse_args()
    Nx = Ny = Nz = None
    single_process_z_range = None
    if curr_rank == 0:
        grid_size = args.gridsize 
        single_process_z_range = args.initrange
        Nx = Ny = Nz = grid_size
        if (Nz % single_process_z_range != 0):
            print("Grid Size in Z direction must be multiple of the initial range of z axis per process")
            comm.Abort(1)

        max_num_process = grid_size // single_process_z_range
        if ((max_num_process & (max_num_process - 1)) != 0):
            print("For simplification, the number of process at the beginning should be power of 2")
            comm.Abort(1)

        current_grid_level = 0
        max_grid_level = int(math.log2(max_num_process))
    
    if curr_rank == 0:
        data = {"Nx": Nx, "Ny": Ny, "Nz": Nz, "single_process_z_range": single_process_z_range, "max_num_process": max_num_process, "current_grid_level" : current_grid_level, "max_grid_level" : max_grid_level}
    else:
        data = None
    
    data = comm.bcast(data, root=0)
    Nx, Ny, Nz = data["Nx"], data["Ny"], data["Nz"]
    max_num_process = data["max_num_process"]
    current_grid_level = data["current_grid_level"]
    max_grid_level = data["max_grid_level"]
    single_process_z_range = data["single_process_z_range"]
    local_approximate_sol = np.zeros((single_process_z_range, Ny, Nx))
    local_r_list = [None] * (max_grid_level + 1)
    if(curr_rank < max_num_process):
        process_activate_flag = True
    else:
        process_activate_flag = False
    
    if setup_debug:
        print(f"Process {curr_rank}: Nx={Nx}, Ny={Ny}, Nz={Nz}, single_process_z_range={single_process_z_range}, process_activate_flag={process_activate_flag}, max_num_process={max_num_process}, current_grid_level={current_grid_level}, max_grid_level={max_grid_level}")

def comm3(data, grid_level):
    ''' periodic boundary. Also send the boundary data to nearby process '''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_z

    if process_activate_flag == False:
        return
    Nz, Ny, Nx = data.shape
    # we always assume the data has shape Nx * Ny * (2 + single_process_z_range)

    # Exchange x-axis data
    data[1:(Nz-1), 1:(Ny-1), Nx - 1] = data[1:(Nz-1), 1:(Ny-1), 1]
    data[1:(Nz-1), 1:(Ny-1), 0]      = data[1:(Nz-1), 1:(Ny-1), Nx - 2]

    # Exchange y-axis data
    data[1:(Nz-1), Ny - 1, :] = data[1:(Nz-1), 1, :]
    data[1:(Nz-1), 0, :] = data[1:(Nz-1), Ny - 2, :]

    # Send the boundary data to nearby process
    send_buf_top = np.copy(data[-2, :, :])
    send_buf_bottom = np.copy(data[1, :, :])
    recv_buf_top = np.empty((Ny, Nx))
    recv_buf_bottom = np.empty((Ny, Nx))
    
    curr_rank = comm.Get_rank()
    process_step = 2 ** grid_level
    index_of_active_process = curr_rank // process_step
    boundary_of_maximum_process = max_num_process // process_step
    if index_of_active_process % 2 == 0:  # Even rank: Send first, then receive
        if index_of_active_process != boundary_of_maximum_process - 1:
            comm.send(send_buf_top, dest= curr_rank + process_step, tag=0)
        else:
            comm.send(send_buf_top, dest= 0, tag=0)
        if index_of_active_process != 0:
            comm.send(send_buf_bottom, dest= curr_rank - process_step, tag=1)
        else:
            comm.send(send_buf_bottom, dest= (boundary_of_maximum_process - 1) * process_step, tag=1)

        # Receive
        if index_of_active_process != 0:
            recv_buf_bottom = comm.recv(source = curr_rank - process_step, tag=0)
        else:
            recv_buf_bottom = comm.recv(source = (boundary_of_maximum_process - 1) * process_step, tag=0)
        if index_of_active_process != boundary_of_maximum_process - 1:
            recv_buf_top = comm.recv(source= curr_rank + process_step, tag=1)
        else:
            recv_buf_top = comm.recv(source= 0, tag=1)

    else: # Odd rank: Receive first, then send
        if index_of_active_process != 0:
            recv_buf_bottom = comm.recv(source = curr_rank - process_step, tag=0)
        else:
            recv_buf_bottom = comm.recv(source = (boundary_of_maximum_process - 1) * process_step, tag=0)
        if index_of_active_process != boundary_of_maximum_process - 1:
            recv_buf_top = comm.recv(source= curr_rank + process_step, tag=1)
        else:
            recv_buf_top = comm.recv(source= 0, tag=1)

        # Send
        if index_of_active_process != boundary_of_maximum_process - 1:
            comm.send(send_buf_top, dest= curr_rank + process_step, tag=0)
        else:
            comm.send(send_buf_top, dest= 0, tag=0)
        if index_of_active_process != 0:
            comm.send(send_buf_bottom, dest= curr_rank - process_step, tag=1)
        else:
            comm.send(send_buf_bottom, dest= (boundary_of_maximum_process - 1) * process_step, tag=1)
    data[0, :, :] = recv_buf_bottom
    data[-1, :, :] = recv_buf_top

        
def residue(u,v,a,grid_level):
    ''' calculate the residue: r = v - Au '''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_z

    if process_activate_flag == False:
        return None
    Nz, Ny, Nx = v.shape
    
    u1 = u[1:(Nz-1), 0:(Ny-2), :] + u[1:(Nz-1), 2:Ny, :] + u[0:(Nz-2), 1:(Ny-1), :] + u[2:Nz, 1:(Ny-1), :]
    u2 = u[0:(Nz-2), 0:(Ny-2), :] + u[0:(Nz-2), 2:Ny, :] + u[2:Nz, 0:(Ny-2), :] +u[2:Nz, 2:Ny, :]
    ua1 = u[1:(Nz-1), 1:(Ny-1), 0:(Nx-2)] + u[1:(Nz-1), 1:(Ny-1), 2:Nx] + u1[:, :, 1:(Nx-1)]
    ua2 = u2[:,:,1:(Nx-1)] + u1[:, :, 0:(Nx-2)] + u1[:, :, 2:Nx]
    ua3 = u2[:,:,0:(Nx-2)] + u2[:,:,2:Nx]
    
    r = np.zeros_like(v)
    r[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] = v[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[0] * u[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[1] * ua1 - a[2] * ua2 - a[3] * ua3
    '''
    r = np.zeros_like(v)
    for i3 in range(1,Nz-1):
        for i2 in range(1,Ny-1):
            u1 = np.zeros((Nx))
            u2 = np.zeros((Nx))
            for i1 in range(Nx):
                u1[i1] = u[i3,i2-1,i1] + u[i3,i2+1,i1] + u[i3-1,i2,i1] + u[i3+1,i2,i1]
                u2[i1] = u[i3-1,i2-1,i1] + u[i3-1,i2+1,i1] + u[i3+1,i2-1,i1] + u[i3+1,i2+1,i1]
            for i1 in range(1,Nx-1):
                r[i3,i2,i1] = v[i3,i2,i1] - a[0] * u[i3,i2,i1] - a[2] * (u2[i1] + u1[i1-1] + u1[i1+1]) - a[3] * (u2[i1-1] + u2[i1+1])
    '''
    comm3(r,grid_level)
    return r 

def rprj3(r, grid_level):
    ''' 
        r : residue in fine grid
        grid_level : current grid level
        s : residue in coarse grid. This is return value

        this function implements the restriction. The number of active process should be halfed after the function
    '''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_z
    if process_activate_flag == False:
        return None

    curr_rank = comm.Get_rank()
    process_step = 2 ** grid_level
    index_of_active_process = curr_rank // process_step
    boundary_of_maximum_process = max_num_process // process_step
    
    if index_of_active_process % 2 == 0: # even rank. Should be active after this function
        r2 = comm.recv(source = curr_rank + process_step, tag=0)
        Nz, Ny, Nx = r.shape
        NNz = Nz
        NNy = Ny // 2 + 1
        NNx = Nx // 2 + 1
        rt = np.concatenate((r[:(Nz-1), :, :], r2[1:, :, :]), axis = 0)
        
        j3 = np.arange(1, NNz-1)
        j2 = np.arange(1, NNy-1)
        j1 = np.arange(1, NNx) # special here
        j0 = np.arange(1, NNx-1)
        i3 = 2 * j3 - 1
        i2 = 2 * j2 - 1
        i1 = 2 * j1 - 1
        i0 = 2 * j0 - 1
        
        ii0 = np.arange(Nx)
        x1 = rt[np.ix_(i3+1, i2, ii0)] + rt[np.ix_(i3+1,i2+2,ii0)] + rt[np.ix_(i3,i2+1,ii0)] + rt[np.ix_(i3+2,i2+1,ii0)]
        y1 = rt[np.ix_(i3, i2, ii0)] + rt[np.ix_(i3+2,i2,ii0)] + rt[np.ix_(i3,i2+2,ii0)] + rt[np.ix_(i3+2,i2+2,ii0)]

        y2 = rt[np.ix_(i3, i2, i0+1)] + rt[np.ix_(i3+2,i2,i0+1)] + rt[np.ix_(i3,i2+2,i0+1)] + rt[np.ix_(i3+2,i2+2,i0+1)]
        x2 = rt[np.ix_(i3+1, i2, i0+1)] + rt[np.ix_(i3+1,i2+2,i0+1)] + rt[np.ix_(i3,i2+1,i0+1)] + rt[np.ix_(i3+2,i2+1,i0+1)]
        
        t2 = rt[np.ix_(i3+1, i2+1, i0)] + rt[np.ix_(i3+1, i2+1, i0+2)] + x2 
        t3 = x1[:,:,i0] + x1[:,:,i0+2] + y2
        t4 = y1[:,:,i0] + y1[:,:,i0+2]

        s = np.zeros((NNz, NNy, NNx), dtype = r.dtype)
        s[1:-1, 1:-1, 1:-1] = 0.5 * (rt[np.ix_(i3+1, i2+1, i0+1)]) + 0.25 * t2 + 0.125 * t3 + 0.0625 * t4

        comm3(s,grid_level+1)
        return s

    else: # odd rank. should be inactive after this function
        comm.send(r, dest= curr_rank - process_step, tag=0)
        process_activate_flag = False
        return None

def interp(z,grid_level):
    ''' interpolate the grid value from coarse grid to fine grid. The function should activate some sleeping process '''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z

    curr_rank = comm.Get_rank()
    process_step = 2 ** grid_level
    index_of_active_process = curr_rank // process_step
    boundary_of_maximum_process = max_num_process // process_step

    fine_process_step = (2 ** (grid_level-1))
    
    if process_activate_flag == True:
        # interpolate data and send the data to target inactive process
        Nz, Ny, Nx = z.shape
        if grid_level == 1:
            # it means we need to update local approximate solution
            next_u = comm.recv(source = curr_rank + fine_process_step, tag=1)
            u = np.concatenate((local_approximate_sol[:(Nz-1), :, :], next_u[1:, :, :]), axis = 0)
        else:
            u = np.zeros((2*Nz-2, 2*Ny-2,2*Nx-2))
        z1 = z[0:(Nz-1), 1:Ny, :] +  z[0:(Nz-1), 0:(Ny-1), :]
        z2 = z[1:Nz, 0:(Ny-1), :] +  z[0:(Nz-1), 0:(Ny-1), :]
        z3 = z[1:Nz, 1:Ny, :] + z[1:Nz, 0:(Ny-1), :] + z1
        
        i3 = np.arange(0,Nz-1)
        i2 = np.arange(0,Ny-1)
        i1 = np.arange(0,Nx-1)
        u[np.ix_(2*i3, 2*i2, 2*i1)] = u[np.ix_(2*i3, 2*i2, 2*i1)] + z[np.ix_(i3, i2, i1)]
        u[np.ix_(2*i3, 2*i2, 2*i1+1)] = u[np.ix_(2*i3, 2*i2, 2*i1+1)] + 0.5*(z[np.ix_(i3, i2, i1+1)] + z[np.ix_(i3, i2, i1)])
        u[np.ix_(2*i3, 2*i2+1, 2*i1)] = u[np.ix_(2*i3, 2*i2+1, 2*i1)] + 0.5 * z1[:,:,0:(Nx-1)]
        u[np.ix_(2*i3, 2*i2+1, 2*i1+1)] = u[np.ix_(2*i3, 2*i2+1, 2*i1+1)] + 0.25 * (z1[:,:,0:(Nx-1)] + z1[:,:,1:Nx])
        u[np.ix_(2*i3+1, 2*i2, 2*i1)] = u[np.ix_(2*i3+1, 2*i2, 2*i1)] + 0.5 * z2[:,:,0:(Nx-1)]
        u[np.ix_(2*i3+1, 2*i2, 2*i1+1)] = u[np.ix_(2*i3+1, 2*i2, 2*i1+1)] + 0.25 * (z2[:,:,0:(Nx-1)] + z2[:,:,1:Nx])
        u[np.ix_(2*i3+1, 2*i2+1, 2*i1)] = u[np.ix_(2*i3+1, 2*i2+1, 2*i1)] + 0.25 * z3[:,:,0:(Nx-1)]
        u[np.ix_(2*i3+1, 2*i2+1, 2*i1+1)] = u[np.ix_(2*i3+1, 2*i2+1, 2*i1+1)] + 0.125*(z3[:,:,0:(Nx-1)] + z3[:,:,1:Nx])
        
        send_u = u[Nz-2:, :, :]
        comm.send(send_u, dest= curr_rank + fine_process_step, tag=0)
        ret = u[:Nz, :, :]
        # comm3(ret, grid_level - 1)
        return ret
    else:
        # check whether current process should be activated or not
        if (curr_rank % fine_process_step == 0):
            if grid_level == 1:
                comm.send(local_approximate_sol, dest= curr_rank - fine_process_step, tag=1)
            process_activate_flag = True
            data = comm.recv(source = curr_rank - fine_process_step, tag=0)
            # comm3(data, grid_level - 1)
            return data
        else:
            return None

def psinv(r, u, c, grid_level):
    ''' apply smoother to the data'''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z
    if process_activate_flag == False:
        return
    
    Nz, Ny, Nx = r.shape

    r1 = r[1:(Nz-1),0:(Ny-2),:] + r[1:(Nz-1),2:Ny,:] + r[0:(Nz-2),1:(Ny-1),:] + r[2:Nz,1:(Ny-1),:]
    r2 = r[0:(Nz-2),0:(Ny-2),:] + r[0:(Nz-2),2:Ny,:] + r[2:Nz,0:(Ny-2),:] + r[2:Nz,2:Ny,:]
    c1 = r[1:(Nz-1),1:(Ny-1),0:(Nx-2)] + r[1:(Nz-1),1:(Ny-1),2:Nx] + r1[:,:,1:(Nx-1)]
    c2 = r2[:,:,1:(Nx-1)] + r1[:,:,0:(Nx-2)] + r1[:,:,2:Nx]
    c3 = r2[:,:,0:(Nx-2)] + r2[:,:,2:Nx]
    u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] = u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[0] * r[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[1] * c1 + c[2] * c2 + c[3] * c3
    
    comm3(u,grid_level)
    
    return

def randlc(x,a):
    r23 = 2.0 ** -23
    r46 = r23 ** 2
    t23 = 2.0 ** 23
    t46 = t23 ** 2

    # Break A into two parts: A = 2^23 * A1 + A2
    t1 = r23 * a
    a1 = int(t1)
    a2 = a - t23 * a1

    # Break X into two parts: X = 2^23 * X1 + X2
    t1 = r23 * x[0]
    x1 = int(t1)
    x2 = x[0] - t23 * x1

    # Compute Z = A1 * X2 + A2 * X1 (mod 2^23)
    t1 = a1 * x2 + a2 * x1
    t2 = int(r23 * t1)
    z = t1 - t23 * t2

    # Compute X = 2^23 * Z + A2 * X2 (mod 2^46)
    t3 = t23 * z + a2 * x2
    t4 = int(r46 * t3)
    x[0] = t3 - t46 * t4

    # Return the normalized random number
    return r46 * x[0]

def power(a,n):
    power = [1.0]
    nj = n
    aj = a

    while nj != 0:
        if (nj % 2) == 1:
            randlc(power, aj)
        randlc([aj], aj)
        nj = nj // 2

    return power[0]

def vranlc(n, x_seed, a, y):
    r23 = 2.0 ** -23
    r46 = r23 ** 2
    t23 = 2.0 ** 23
    t46 = t23 ** 2

    t1 = r23 * a
    a1 = int(t1)
    a2 = a - t23 * a1
    x = x_seed[0]

    for i in range(n):
        t1 = r23 * x
        x1 = int(t1)
        x2 = x - t23 * x1

        t1 = a1 * x2 + a2 * x1
        t2 = int(r23 * t1)
        z = t1 - t23 * t2

        t3 = t23 * z + a2 * x2
        t4 = int(r46 * t3)
        x = t3 - t46 * t4

        y[i] = r46 * x

    x_seed[0] = x

def zran3(z, grid_level, x_seed, a):
    ''' generate random right side data'''
    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z

    rank = comm.Get_rank()
    Nz, Ny, Nx = z.shape
    for i3 in range(1,(Nz-1)):
        for i2 in range(1,(Ny-1)):
            vranlc(Nx-2, x_seed, a, z[i3, i2, 1:(Nx-1)])
    
    ten_max = [(0.0, (0, 0, 0))] * 10
    ten_min = [(100.0, (0, 0, 0))] * 10

    for i3 in range(1,(Nz-1)):
        for i2 in range(1,(Ny-1)):
            for i1 in range(1,(Nx-1)):
                value = z[i3, i2, i1]
                if value > ten_max[0][0]:
                    ten_max[0] = (value, (i3 - 1 + rank * single_process_z_range, i2, i1))
                    ten_max.sort(key=lambda x: x[0])
                if value < ten_min[0][0]:
                    ten_min[0] = (value, (i3 - 1 + rank * single_process_z_range, i2, i1))
                    ten_min.sort(key=lambda x: -x[0])
    
    gathered_max_arrays = comm.gather(ten_max, root=0)
    gathered_min_arrays = comm.gather(ten_min, root=0)

    if (rank == 0):
        gathered_max_arrays = [item for sublist in gathered_max_arrays for item in sublist]
        gathered_min_arrays = [item for sublist in gathered_min_arrays for item in sublist]
        sorted_max = sorted(gathered_max_arrays, key=lambda x: x[0])
        top_max_arrays = sorted_max[-10:]
        sorted_min = sorted(gathered_min_arrays, key=lambda x: x[0])
        top_min_arrays = sorted_min[:10]
    else:
        top_max_arrays = None
        top_min_arrays = None

    global_max_values = comm.bcast(top_max_arrays, root=0)
    global_min_values = comm.bcast(top_min_arrays, root=0)
    
    for i3 in range(1,(Nz-1)):
        for i2 in range(1,(Ny-1)):
            for i1 in range(1,(Nx-1)):
                global_index = (i3 - 1 + rank * single_process_z_range, i2, i1)
                if (z[i3, i2, i1],global_index) in global_max_values:
                    z[i3, i2, i1] = 1.0
                elif (z[i3, i2, i1],global_index) in global_min_values:
                    z[i3, i2, i1] = -1.0
                else:
                    z[i3, i2, i1] = 0.0

    comm3(z, grid_level)
    return 

def mg3P(v,a,c):
    ''' implement the v-cycle multigrid algorithm '''

    global single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z, local_r_list

    rank = comm.Get_rank()
    for i in range(max_grid_level-1):
        local_r_list[i+1] = rprj3(local_r_list[i],i)
        comm.barrier()
    
    temp_u = np.zeros_like(local_r_list[max_grid_level-1])
    psinv(local_r_list[max_grid_level-1],temp_u,c,max_grid_level-1)
    comm.barrier()
    
    for i in range(max_grid_level-1, 1, -1):
        temp_u = interp(temp_u, i)
        comm.barrier()
        local_r_list[i-1] = residue(temp_u,local_r_list[i-1],a,i-1)
        comm.barrier()
        psinv(local_r_list[i-1],temp_u,c,i-1)
        comm.barrier()

    local_approximate_sol = interp(temp_u,1)
    comm.barrier()
    local_r_list[0] = residue(local_approximate_sol,v,a,0)
    comm.barrier()
    psinv(local_r_list[0],local_approximate_sol,c,0)
    comm.barrier()
    # if process_activate_flag == True:
    #     print(f"Rank {rank}")
    #     print(local_approximate_sol[1,:,:])
    
    return

''' test part '''
def test_comm3(data_z, data_y, data_x, grid_level):
    """
    Creates a 3D numpy array with shape (data_z, data_y, data_x).
    Each element is initialized with a unique value based on its indices for easy verification.
    """
    test_array = np.zeros((data_z, data_y, data_x), dtype=int)
    rank = comm.Get_rank()
    # Initialize array with values based on their indices for easy debugging
    for z in range(data_z):
        for y in range(data_y):
            for x in range(data_x):
                test_array[z, y, x] = (z + rank * 2**grid_level) * 10000 + y * 100 + x

    print(f"Rank {rank}: Array Before comm3:")
    print(test_array)

    comm3(test_array, grid_level)

    print(f"\nRank {rank}: Array After comm3:")
    print(test_array)
    
def test_residue(data_z, data_y, data_x, grid_level):
    test_array = np.zeros((data_z, data_y, data_x), dtype=float)
    rank = comm.Get_rank()
    # Initialize array with values based on their indices for easy debugging
    for z in range(data_z):
        for y in range(data_y):
            for x in range(data_x):
                test_array[z, y, x] = (z + rank * 2**grid_level) * 10000 + y * 100 + x

    u = np.ones((data_z, data_y, data_x), dtype=float)
    a = np.zeros((4), dtype=float)
    
    print(test_array.shape)
    r = residue(u, test_array, a, grid_level)
    print(r.shape)
    
def test_zran3(data_z, data_y, data_x, grid_level):
    rank = comm.Get_rank()
    z = np.zeros((data_z, data_y, data_x), dtype=np.float64)
    x_seed = [314159265.0 + rank]
    a = 5.0 ** 13

    zran3(z, grid_level, x_seed, a)
    print(f"Process {rank}:\n", z[1,:,:])

def get_r_norm(r):
    rank = comm.Get_rank()
    r_norm = np.sum(r[1:-1,1:-1,1:-1]**2)**(0.5)
    recv_data = comm.gather(r_norm,root=0)
    if rank == 0:
        norm_sum = 0.0
        for data in recv_data:
            norm_sum = norm_sum + data
        print(f"Process {rank}:\n", norm_sum, flush = True)

# debug part
def sample_local_z(filename):
    global local_z

    groups = []         # To store each z-level (each a list of rows)
    current_group = []  # To collect rows for the current z-level

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if set(line) == {'_'}:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                continue

            if line.startswith('[') and line.endswith(']'):
                row_str = line[1:-1].strip()
                if row_str:  # If the row is not empty
                    row = [float(x) for x in row_str.split()]
                else:
                    row = []
                current_group.append(row)
    
        if current_group:
            groups.append(current_group)

    arr = np.array(groups)
    
    rank = comm.Get_rank()
    data_level = rank + 1
    local_z = np.zeros((3,10,10))
    local_z[0,:,:] = arr[data_level-1,:,:]
    local_z[1,:,:] = arr[data_level,:,:]
    local_z[2,:,:] = arr[data_level+1,:,:]
    return 

def show_local_z():
    rank = comm.Get_rank()
    print(f"Process {rank}")
    print(local_z[1,:,:])

def main():
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol, local_z, local_r_list
    setup()
    comm.barrier()
    data_x = Nx + 2
    data_y = Ny + 2
    data_z = single_process_z_range + 2

    rank = comm.Get_rank()
    local_z = np.zeros((data_z, data_y, data_x), dtype=np.float64)
    x_seed = [314159265.0 + rank]
    a = 5.0 ** 13
    
    # sample_local_z('sample_z.txt')
    # show_local_z()

    zran3(local_z, 0, x_seed, a)
    comm.barrier()
    a = [-8.0/3.0, 0.0, 1.0/6.0, 1.0/12.0]
    iteration_number = 20

    if Nx != Ny or Nx != Nz:
        smooth_type = "U"
    elif Nx == 32 and iteration_number == 4:
        smooth_type = "S"
    elif Nx == 128 and iteration_number == 4:
        smooth_type = "W"
    elif Nx == 256 and iteration_number == 4:
        smooth_type = "A"
    elif Nx == 256 and iteration_number == 20:
        smooth_type = "B"
    elif Nx == 512 and iteration_number == 20:
        smooth_type = "C"
    elif Nx == 1024 and iteration_number == 50:
        smooth_type = "D"
    elif Nx == 2048 and iteration_number == 50:
        smooth_type = "E"
    else:
        smooth_type = "U"

    if smooth_type == "A" or  smooth_type == "S" or  smooth_type == "W":
        c = [-3.0/8.0, 1.0/32.0, -1.0/64.0, 0.0]
    else:
        c = [-3.0/17.0, 1.0/33.0, -1.0/61.0, 0.0]
    local_approximate_sol = np.zeros_like(local_z)
    local_r_list[0] = residue(local_approximate_sol, local_z, a, 0)
    
    comm.barrier()
    for i in range(iteration_number):
        mg3P(local_z,a,c)
        comm.barrier()
        local_r_list[0] = residue(local_approximate_sol, local_z, a, 0)
        comm.barrier()
        get_r_norm(local_r_list[0])
     
    # test_comm3(data_z, data_y, data_x, 0)
    # test_residue(data_z, data_y, data_x, 0)
    # test_zran3(data_z, data_y, data_x, 0)

if __name__ == "__main__":
    main()



