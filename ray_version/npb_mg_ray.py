import numpy as np
from parser import get_parser
import ray
import ray_mpi
import math

@ray.remote
class mg_actor:
    def __init__(self, world_size, MPIRuntime, gridsize, initrange, itn):
        self.rank = -1
        self.world_size = world_size
        self.MPIRuntime = MPIRuntime
        
        self.Nx = gridsize
        self.Ny = gridsize
        self.Nz = gridsize

        self.single_process_z_range = initrange
        self.iteration_number = itn

    def set_rank(self, rank):
        self.rank = rank

    def setup(self):
        self.max_grid_level = int(math.log2(self.world_size))
        self.local_approximate_sol = np.zeros((self.single_process_z_range, self.Ny, self.Nx))
        self.local_r_list = [None] * (self.max_grid_level + 1)
        self.process_activate_flag = True
        self.setup_debug = False

        if self.setup_debug == True:
            print(f"Process {self.rank}: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, single_process_z_range={self.single_process_z_range}, process_activate_flag={self.process_activate_flag}, max_grid_level={self.max_grid_level}")
    
    def comm3(self, data, grid_level):
        if self.process_activate_flag == False:
            return 
        
        Nz, Ny, Nx = data.shape

        data[1:(Nz-1), 1:(Ny-1), Nx - 1] = data[1:(Nz-1), 1:(Ny-1), 1]
        data[1:(Nz-1), 1:(Ny-1), 0]      = data[1:(Nz-1), 1:(Ny-1), Nx - 2]

        data[1:(Nz-1), Ny - 1, :] = data[1:(Nz-1), 1, :]
        data[1:(Nz-1), 0, :] = data[1:(Nz-1), Ny - 2, :]

        send_buf_top = np.copy(data[-2, :, :])
        send_buf_bottom = np.copy(data[1, :, :])
        recv_buf_top = np.empty((Ny, Nx))
        recv_buf_bottom = np.empty((Ny, Nx))
        
        process_step = 2 ** grid_level
        index_of_active_process = self.rank // process_step
        boundary_of_maximum_process = self.world_size // process_step
        if index_of_active_process % 2 == 0:
            if index_of_active_process != boundary_of_maximum_process - 1:
                ray.get(self.MPIRuntime.send.remote(send_buf_top, src_rank = self.rank, dest_rank = self.rank + process_step))
            else:
                ray.get(self.MPIRuntime.send.remote(send_buf_top, src_rank = self.rank, dest_rank = 0))
            if index_of_active_process != 0:
                ray.get(self.MPIRuntime.send.remote(send_buf_bottom, src_rank = self.rank, dest_rank = self.rank - process_step))
            else:
                ray.get(self.MPIRuntime.send.remote(send_buf_bottom, src_rank = self.rank, dest_rank = (boundary_of_maximum_process - 1) * process_step))
            
            if index_of_active_process != 0:
                recv_buf_bottom = ray.get(self.MPIRuntime.recv.remote(src_rank = self.rank - process_step, dest_rank = self.rank))
            else:
                recv_buf_bottom = ray.get(self.MPIRuntime.recv.remote(src_rank = (boundary_of_maximum_process - 1) * process_step, dest_rank = self.rank))
            if index_of_active_process != boundary_of_maximum_process - 1:
                recv_buf_top = ray.get(self.MPIRuntime.recv.remote(src_rank = self.rank + process_step, dest_rank = self.rank))
            else:
                recv_buf_top = ray.get(self.MPIRuntime.recv.remote(src_rank = 0, dest_rank = self.rank))
        else:
            if index_of_active_process != 0:
                recv_buf_bottom = ray.get(self.MPIRuntime.recv.remote(src_rank = self.rank - process_step, dest_rank = self.rank))
            else:
                recv_buf_bottom = ray.get(self.MPIRuntime.recv.remote(src_rank = (boundary_of_maximum_process - 1) * process_step, dest_rank = self.rank))
            if index_of_active_process != boundary_of_maximum_process - 1:
                recv_buf_top = ray.get(self.MPIRuntime.recv.remote(src_rank = self.rank + process_step, dest_rank = self.rank))
            else:
                recv_buf_top = ray.get(self.MPIRuntime.recv.remote(src_rank = 0, dest_rank = self.rank))

            if index_of_active_process != boundary_of_maximum_process - 1:
                ray.get(self.MPIRuntime.send.remote(send_buf_top, src_rank = self.rank, dest_rank = self.rank + process_step))
            else:
                ray.get(self.MPIRuntime.send.remote(send_buf_top, src_rank = self.rank, dest_rank = 0))
            if index_of_active_process != 0:
                ray.get(self.MPIRuntime.send.remote(send_buf_bottom, src_rank = self.rank, dest_rank = self.rank - process_step))
            else:
                ray.get(self.MPIRuntime.send.remote(send_buf_bottom, src_rank = self.rank, dest_rank = (boundary_of_maximum_process - 1) * process_step))
        
        data[0, :, :] = recv_buf_bottom
        data[-1, :, :] = recv_buf_top
    
    def residue(self, u, v, a, grid_level):
        if self.process_activate_flag == False:
            return None
        
        Nz, Ny, Nx = v.shape

        u1 = u[1:(Nz-1), 0:(Ny-2), :] + u[1:(Nz-1), 2:Ny, :] + u[0:(Nz-2), 1:(Ny-1), :] + u[2:Nz, 1:(Ny-1), :]
        u2 = u[0:(Nz-2), 0:(Ny-2), :] + u[0:(Nz-2), 2:Ny, :] + u[2:Nz, 0:(Ny-2), :] +u[2:Nz, 2:Ny, :]
        ua1 = u[1:(Nz-1), 1:(Ny-1), 0:(Nx-2)] + u[1:(Nz-1), 1:(Ny-1), 2:Nx] + u1[:, :, 1:(Nx-1)]
        ua2 = u2[:,:,1:(Nx-1)] + u1[:, :, 0:(Nx-2)] + u1[:, :, 2:Nx]
        ua3 = u2[:,:,0:(Nx-2)] + u2[:,:,2:Nx]

        r = np.zeros_like(v)
        r[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] = v[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[0] * u[1:(Nz-1), 1:(Ny-1), 1:(Nx-1)] - a[1] * ua1 - a[2] * ua2 - a[3] * ua3

        self.comm3(r,grid_level)

        return r

    def rprj3(self, r, grid_level):
        if self.process_activate_flag == False:
            return None

        process_step = 2 ** grid_level
        index_of_active_process = self.rank // process_step
        boundary_of_maximum_process = self.world_size // process_step
        
        if index_of_active_process % 2 == 0:
            r2 = ray.get(self.MPIRuntime.recv.remote(src_rank = self.rank + process_step, dest_rank = self.rank))
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

            self.comm3(s,grid_level+1)
            return s

        else:
            ray.get(self.MPIRuntime.send.remote(r, src_rank=self.rank, dest_rank=self.rank-process_step))
            self.process_activate_flag = False
            return None
    
    def interp(self,z,grid_level):
        process_step = 2 ** grid_level
        index_of_active_process = self.rank // process_step
        boundary_of_maximum_process = self.world_size // process_step

        fine_process_step = (2 ** (grid_level-1))

        if self.process_activate_flag == True:
            Nz, Ny, Nx = z.shape
            if grid_level == 1:
                next_u = ray.get(self.MPIRuntime.recv.remote(src_rank=self.rank+fine_process_step, dest_rank=self.rank))
                u = np.concatenate((self.local_approximate_sol[:(Nz-1), :, :], next_u[1:, :, :]), axis = 0)
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
            ray.get(self.MPIRuntime.send.remote(send_u, src_rank=self.rank, dest_rank=self.rank+fine_process_step))
            return u[:Nz, :, :]
        else:
            if(self.rank % fine_process_step == 0):
                if grid_level == 1:
                    ray.get(self.MPIRuntime.send.remote(self.local_approximate_sol, src_rank=self.rank, dest_rank=self.rank-fine_process_step))
                self.process_activate_flag = True
                data = ray.get(self.MPIRuntime.recv.remote(src_rank=self.rank-fine_process_step, dest_rank=self.rank))
                return data.copy()
            else:
                return None

    def psinv(self, r,u,c, grid_level):
        if self.process_activate_flag == False:
            return 

        Nz, Ny, Nx = r.shape
        r1 = r[1:(Nz-1),0:(Ny-2),:] + r[1:(Nz-1),2:Ny,:] + r[0:(Nz-2),1:(Ny-1),:] + r[2:Nz,1:(Ny-1),:]
        r2 = r[0:(Nz-2),0:(Ny-2),:] + r[0:(Nz-2),2:Ny,:] + r[2:Nz,0:(Ny-2),:] + r[2:Nz,2:Ny,:]
        c1 = r[1:(Nz-1),1:(Ny-1),0:(Nx-2)] + r[1:(Nz-1),1:(Ny-1),2:Nx] + r1[:,:,1:(Nx-1)]
        c2 = r2[:,:,1:(Nx-1)] + r1[:,:,0:(Nx-2)] + r1[:,:,2:Nx]
        c3 = r2[:,:,0:(Nx-2)] + r2[:,:,2:Nx]
        u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] = u[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[0] * r[1:(Nz-1),1:(Ny-1),1:(Nx-1)] + c[1] * c1 + c[2] * c2 + c[3] * c3

        self.comm3(u,grid_level)
        return

    def randlc(self, x, a):
        r23 = 2.0 ** -23
        r46 = r23 ** 2
        t23 = 2.0 ** 23
        t46 = t23 ** 2

        t1 = r23 * a
        a1 = int(t1)
        a2 = a - t23 * a1
        
        t1 = r23 * x[0]
        x1 = int(t1)
        x2 = x[0] - t23 * x1

        t1 = a1 * x2 + a2 * x1
        t2 = int(r23 * t1)
        z = t1 - t23 * t2

        t3 = t23 * z + a2 * x2
        t4 = int(r46 * t3)
        x[0] = t3 - t46 * t4

        return r46 * x[0]

    def power(self,a,n):
        power = [1.0]
        nj = n
        aj = a

        while nj != 0:
            if (nj % 2) == 1:
                self.randlc(power, aj)
            self.randlc([aj], aj)
            nj = nj // 2

        return power[0]

    def vranlc(self, n, x_seed, a, y):
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
    
    def zran3(self, z, grid_level, x_seed, a):
        Nz, Ny, Nx = z.shape
        for i3 in range(1,(Nz-1)):
            for i2 in range(1,(Ny-1)):
                self.vranlc(Nx-2, x_seed, a, z[i3, i2, 1:(Nx-1)])
        
        ten_max = [(0.0, (0, 0, 0))] * 10
        ten_min = [(100.0, (0, 0, 0))] * 10

        for i3 in range(1,(Nz-1)):
            for i2 in range(1,(Ny-1)):
                for i1 in range(1,(Nx-1)):
                    value = z[i3, i2, i1]
                    if value > ten_max[0][0]:
                        ten_max[0] = (value, (i3 - 1 + self.rank * self.single_process_z_range, i2, i1))
                        ten_max.sort(key=lambda x: x[0])
                    if value < ten_min[0][0]:
                        ten_min[0] = (value, (i3 - 1 + self.rank * self.single_process_z_range, i2, i1))
                        ten_min.sort(key=lambda x: -x[0])

        gathered_max_arrays = ray.get(self.MPIRuntime.gather.remote(self.rank, ten_max, root_rank=0))
        gathered_min_arrays = ray.get(self.MPIRuntime.gather.remote(self.rank, ten_min, root_rank=0))

        if (self.rank == 0):
            gathered_max_arrays = [item for sublist in gathered_max_arrays for item in sublist]
            gathered_min_arrays = [item for sublist in gathered_min_arrays for item in sublist]
            sorted_max = sorted(gathered_max_arrays, key=lambda x: x[0])
            top_max_arrays = sorted_max[-10:]
            sorted_min = sorted(gathered_min_arrays, key=lambda x: x[0])
            top_min_arrays = sorted_min[:10]
        else:
            top_max_arrays = None
            top_min_arrays = None

        global_max_values = ray.get(self.MPIRuntime.broadcast.remote(self.rank,top_max_arrays,0))
        global_min_values = ray.get(self.MPIRuntime.broadcast.remote(self.rank,top_min_arrays,0))

        for i3 in range(1, (Nz-1)):
            for i2 in range(1, (Ny-1)):
                for i1 in range(1,(Nx-1)):
                    global_index = (i3 - 1 + self.rank * self.single_process_z_range, i2, i1)
                    if (z[i3, i2, i1],global_index) in global_max_values:
                        z[i3, i2, i1] = 1.0
                    elif (z[i3, i2, i1],global_index) in global_min_values:
                        z[i3, i2, i1] = -1.0
                    else:
                        z[i3, i2, i1] = 0.0

        self.comm3(z,grid_level)
        return

    def mg3P(self,v,a,c):
        for i in range(self.max_grid_level-1):
            self.local_r_list[i+1] = self.rprj3(self.local_r_list[i],i)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))

        temp_u = np.zeros_like(self.local_r_list[self.max_grid_level-1])
        self.psinv(self.local_r_list[self.max_grid_level-1],temp_u,c,self.max_grid_level-1)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        for i in range(self.max_grid_level-1, 1, -1):
            temp_u = self.interp(temp_u, i)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))
            self.local_r_list[i-1] = self.residue(temp_u,self.local_r_list[i-1],a,i-1)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))
            self.psinv(self.local_r_list[i-1],temp_u,c,i-1)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))

        self.local_approximate_sol = self.interp(temp_u,1)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        self.local_r_list[0] = self.residue(self.local_approximate_sol,v,a,0)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        self.psinv(self.local_r_list[0],self.local_approximate_sol,c,0)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))

        return

    # Test Parts
    def test_comm3(self, data_z, data_y, data_x, grid_level):
        test_array = np.zeros((data_z, data_y, data_x), dtype=int)
        rank = self.rank
        # Initialize array with values based on their indices for easy debugging
        for z in range(data_z):
            for y in range(data_y):
                for x in range(data_x):
                    test_array[z, y, x] = (z + rank * 2**grid_level) * 10000 + y * 100 + x

        print(f"Rank {rank}: Array Before comm3:")
        print(test_array)
        
        self.comm3(test_array, grid_level)

        print(f"\nRank {rank}: Array After comm3:")
        print(test_array)

    def get_r_norm(self,r):
        r_norm = np.sum(r[1:-1,1:-1,1:-1]**2)**(0.5)
        recv_data = ray.get(self.MPIRuntime.gather.remote(self.rank, r_norm, 0))
        if self.rank == 0:
            norm_sum = 0.0
            for data in recv_data:
                norm_sum = norm_sum + data
            print(f"Process {self.rank}:\n", norm_sum, flush = True)

    def eval(self):
        self.setup()
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        data_x = self.Nx + 2
        data_y = self.Ny + 2
        data_z = self.single_process_z_range + 2

        self.local_z = np.zeros((data_z, data_y, data_x), dtype=np.float64)
        x_seed = [314159265.0 + self.rank]
        a = 5.0 ** 13
        
        self.zran3(self.local_z, 0, x_seed, a)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        a = [-8.0/3.0, 0.0, 1.0/6.0, 1.0/12.0]
        
        if self.Nx != self.Ny or self.Nx != self.Nz:
            smooth_type = "U"
        elif self.Nx == 32 and self.iteration_number == 4:
            smooth_type = "S"
        elif self.Nx == 128 and self.iteration_number == 4:
            smooth_type = "W"
        elif self.Nx == 256 and self.iteration_number == 4:
            smooth_type = "A"
        elif self.Nx == 256 and self.iteration_number == 20:
            smooth_type = "B"
        elif self.Nx == 512 and self.iteration_number == 20:
            smooth_type = "C"
        elif self.Nx == 1024 and self.iteration_number == 50:
            smooth_type = "D"
        elif self.Nx == 2048 and self.iteration_number == 50:
            smooth_type = "E"
        else:
            smooth_type = "U"

        if smooth_type == "A" or  smooth_type == "S" or  smooth_type == "W":
            c = [-3.0/8.0, 1.0/32.0, -1.0/64.0, 0.0]
        else:
            c = [-3.0/17.0, 1.0/33.0, -1.0/61.0, 0.0]
        
        self.local_approximate_sol = np.zeros_like(self.local_z)
        self.local_r_list[0] = self.residue(self.local_approximate_sol, self.local_z, a, 0)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))

        for i in range(self.iteration_number):
            self.mg3P(self.local_z,a,c)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))
            self.local_r_list[0] = self.residue(self.local_approximate_sol, self.local_z, a, 0)
            ray.get(self.MPIRuntime.barrier.remote(self.rank))
            self.get_r_norm(self.local_r_list[0])
        # self.test_comm3(data_z, data_y, data_x, 0)

def main():
    RayMPIRuntime = ray_mpi.RayMPIRuntime.remote()
    worker_list = []
    args = get_parser().parse_args()
    
    grid_size = args.gridsize
    single_process_z_range = args.initrange
    nit = args.itn
    if (grid_size % single_process_z_range != 0):
        print("Grid Size in Z direction must be multiple of the initial range of z axis per process")
        sys.exit(1)

    max_num_process = grid_size // single_process_z_range
    if ((max_num_process & (max_num_process - 1)) != 0):
        print("For simplification, the number of process at the beginning should be power of 2")
        sys.exit(1)
    
    world_size = max_num_process

    for i in range(world_size):
        single_worker = mg_actor.remote(world_size, RayMPIRuntime, grid_size, single_process_z_range, nit)
        worker_list.append(single_worker)
    ray.get(RayMPIRuntime.init.remote(worker_list, [i for i in range(world_size)]))

    work_list = []
    for worker in worker_list:
        work_list.append(worker.eval.remote())
    ray.get(work_list)

if __name__ == "__main__":
    main()
