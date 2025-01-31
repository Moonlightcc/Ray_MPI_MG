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

def setup():
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol

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

    if(curr_rank < max_num_process):
        process_activate_flag = True
    else:
        process_activate_flag = False
    
    if setup_debug:
        print(f"Process {curr_rank}: Nx={Nx}, Ny={Ny}, Nz={Nz}, single_process_z_range={single_process_z_range}, process_activate_flag={process_activate_flag}, max_num_process={max_num_process}, current_grid_level={current_grid_level}, max_grid_level={max_grid_level}")

def comm3(data, grid_level):
    ''' periodic boundary. Also send the boundary data to nearby process '''
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag

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
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag

    if process_activate_flag == False:
        return None
    Nz, Ny, Nx = u.shape

    u1 = u[1:(Nz-1), 0:(Ny-2), :] + u[1:(Nz-1), 2:Ny, :] + u[0:(Nz-2), 1:(Ny-1), :] + u[2:Nz, 1:(Ny-1), :]
    u2 = u[0:(Nz-2), 0:(Ny-2), :] + u[0:(Nz-2), 2:Ny, :] + u[2:Nz, 0:(Ny-2), :] +u[2:Nz, 2:Ny, :]
    ua1 = u[1:(Nz-1), 1:(Ny-1), 0:(Nx-2)] + u[1:(Nz-1), 1:(Ny-1), 2:Nx] + u1[:, :, 1:(Nx-1)]
    ua2 = u2[:,:,1:(Nx-1)] + u1[:, :, 0:(Nx-2)] + u1[:, :, 2:Nx]
    ua3 = u2[:,:,0:(Nx-2)] + u2[:,:,2:Nx]
    
    r = np.zeros_like(u)
    r[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] = v[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[0] * u[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[1] * ua1 - a[2] * ua2 - a[3] * ua3

    # for i3 in range(1,Nz-1):
    #     for i2 in range(1, Ny-1):
    #         u1 = u[i3,i2-1,:] + u[i3,i2+1,:] + u[i3-1,i2,:] + u[i3+1,i2,:]
    #         u2 = u[i3-1,i2-1,:] + u[i3-1,i2+1,:] + u[i3+1,i2-1,:] + u[i3+1,i2+1,:]
    #         r[i3,i2,1:(Nz-1)] = v[i3,i2,1:(Nz-1)] - a[0] * u[i3,i2,1:(Nz-1)] - a[1] * (u[i1-1,i2,1:(Nz-1)] + u[i1+1,i2,1:(Nz-1)] + u1[1:(Nz-1)]) - a[2] * (u2[1:(Nz-1)] + u1[0:(Nz-2)] + u1[2:Nz]) - a[3] * (u2[0:(Nz-2)] + u2[2:Nz])

    comm3(r,grid_level)
    return r

def rprj3(r, grid_level):
    ''' 
        r : residue in fine grid
        grid_level : current grid level
        s : residue in coarse grid. This is return value

        this function implements the restriction. The number of active process should be halfed after the function
    '''
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag
    if process_activate_flag == False:
        return None

    curr_rank = comm.Get_rank()
    process_step = 2 ** grid_level
    index_of_active_process = curr_rank // process_step
    boundary_of_maximum_process = max_num_process // process_step
    
    if index_of_active_process % 2 == 0: # even rank. Should be active after this function
        r2 = comm.recv(source = curr_rank + process_step, tag=0)
        Nz, Ny, Nx = r.shape
        rt = np.concatenate((r[:(Nz-1), :, :], r2[1:, :, :]), axis = 0)
        
        j3 = np.arange(1, Nz-1)
        j2 = np.arange(1, Ny-1)
        j1 = np.arange(1, Nx) # special here
        j0 = np.arange(1, Nx-1)
        i3 = 2 * j3 - 1
        i2 = 2 * j2 - 1
        i1 = 2 * j1 - 1
        i0 = 2 * j0 - 1

        x1 = rt[np.ix_(i3+1, i2, i1)] + rt[np.ix_(i3+1,i2+2,i1)] + rt[np.ix_(i3,i2+1,i1)] + rt[np.ix_(i3+2,i2+1,i1)]
        y1 = rt[np.ix_(i3, i2, i1)] + rt[np.ix_(i3+2,i2,i1)] + rt[np.ix_(i3,i2+2,i1)] + rt[np.ix_(i3+2,i2+2,i1)]

        y2 = rt[np.ix_(i3, i2, i0+1)] + rt[np.ix_(i3+2,i2,i0+1)] + rt[np.ix_(i3,i2+2,i0+1)] + rt[np.ix_(i3+2,i2+2,i0+1)]
        x2 = rt[np.ix_(i3+1, i2, i0+1)] + rt[np.ix_(i3+1,i2+2,i0+1)] + rt[np.ix_(i3,i2+1,i0+1)] + rt[np.ix_(i3+2,i2+1,i0+1)]
        
        t2 = rt[np.ix_(i3+1, i2+1, i0)] + rt[np.ix_(i3+1, i2+1, i0+2)] + x2
        t3 = x1[:,:,i0] + x1[:,:,i0+2] + y2
        t4 = y1[:,:,i0] + y1[:,:,i0+2]

        s = np.zeros_like(r)
        s[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] = 0.5 * (rt[np.ix_(i3+1, i2+1, i0+1)]) + 0.25 * t2 + 0.125 * t3 + 0.0625 * t4

        comm3(s,grid_level+1)
        return s

    else: # odd rank. should be inactive after this function
        comm.send(r, dest= curr_rank - process_step, tag=0)
        process_activate_flag = False
        return None

def interp(z,grid_level):
    ''' interpolate the grid value from coarse grid to fine grid. The function should activate some sleeping process '''
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol

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
            u = np.concatenate((local_approximate_sol[:(Nz-1), :, :], r2[next_u:, :, :]), axis = 0)
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
        
        send_u = u[Nx-2, :, :]
        comm.send(send_u, dest= curr_rank + fine_process_step, tag=0)
        return u[:Nz, :, :]
    else:
        # check whether current process should be activated or not
        if (curr_rank % fine_process_step == 0):
            if grid_level == 1:
                comm.send(local_approximate_sol, dest= curr_rank - fine_process_step, tag=1)
            process_activate_flag = True
            data = comm.recv(source = curr_rank - fine_process_step, tag=0)
            return data
        else:
            return None

def psinv(r, u, c, grid_level):
    ''' apply smoother to the data'''
    global Nx, Ny, Nz, single_process_z_range, max_num_process, current_grid_level, max_grid_level, process_activate_flag, local_approximate_sol
    if process_activate_flag == False:
        return
    
    r1 = r[1:(Nz-1),0:(Ny-2),:] + r[1:(Nz-1),2:Ny,:] + r[0:(Nz-2),1:(Ny-1),:] + r[2:Nz,1:(Ny-1),:]
    r2 = r[0:(Nz-2),0:(Ny-2),:] + r[0:(Nz-2),2:Ny,:] + r[2:Nz,0:(Ny-2),:] + r[2:Nz,2:Ny,:]
    c1 = r[1:(Nz-1),1:(Ny-1),0:(Nx-2)] + r[1:(Nz-1),1:(Ny-1),2:Nx] + r1[:,:,1:(Nx-1)]
    c2 = r2[:,:,1:(Nx-1)] + r1[:,:,0:(Nx-2)] + r1[:,:,2:Nx]
    c3 = r2[:,:,0:(Nx-2)] + r2[:,:,2:Nx]
    u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] = u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[0] * r[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[1] * c1 + c[2] * c2 + c[3] * c3

    comm3(u,grid_level)

    return

def zran3():
    ''' generate random right side data'''
    # TODO
    return 

def mg3P():
    ''' implement the v-cycle multigrid algorithm '''
    # TODO
    # Au = v
    zran3() generate v
    
    max_iteration_number
    for i in range(5):
        # shrink once in grid
        residue()
        rrprj3()
    
    psinv() get solution in the coarsest grid
    for i in range(5):
        # expand rank after iteration
        interp()
        psinv()

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
    


def main():
    setup()
    data_x = Nx + 2
    data_y = Ny + 2
    data_z = single_process_z_range + 2
    # test_comm3(data_z, data_y, data_x, 0)
    # test_residue(data_z, data_y, data_x, 0)
    
if __name__ == "__main__":
    main()



