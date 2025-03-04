import ray
# import ray.util.collective as col
import ray_mpi
import time

ray.init(address="auto")

@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, MPIRuntime):
        self.rank = -1
        self.word_size = 3
        self.MPIRuntime = MPIRuntime
    
    def set_rank(self, rank):
        self.rank = rank

    # def send(
    #     self,
    #     data,
    #     src_rank: int,
    #     dest_rank: int
    # ):
    #     data_ref = ray.put(data)
    #     signal = ray.get(self.MPIRuntime.send.remote(data_ref, src_rank, dest_rank))
    #     ray.get(signal.wait.remote(2))
    
    # def recv(
    #     self,
    #     src_rank: int,
    #     dest_rank: int
    # ):
    #     signal = ray.get(self.MPIRuntime.wait_recv.remote(src_rank, dest_rank))
    #     ray.get(signal.wait.remote(2))

    #     return ray.get(self.MPIRuntime.recv.remote(src_rank, dest_rank))
        

    def prefix_sum(self, data):
        # print("My rank is {}".format(self.rank))

        for i in range(1, len(data)):
            data[i] = data[i] + data[i - 1]
        
        previous_sum = 0
        if self.rank > 0:
            previous_sum = ray.get(self.MPIRuntime.recv.remote(self.rank-1, self.rank))

        next_sum = previous_sum + data[-1]
        
        if self.rank < self.word_size - 1:
            ray.get(self.MPIRuntime.send.remote(next_sum, self.rank, self.rank + 1))
        
        for i in range(len(data)):
            data[i] += previous_sum

    def main(self):
        # the main work of the rank
        # print("My rank is {}".format(self.rank))
        # time.sleep(5)


        # self.prefix_sum(data)
        if self.rank == 0:
            time.sleep(10)
        print("haime caonima", flush = True)
        ray.get(self.MPIRuntime.barrier.remote(self.rank))
        print("caonima", flush = True)
        # time.sleep(1)

        # if self.rank == 0:
        #     # time.sleep(3)
        #     ray.get(self.MPIRuntime.send.remote("hello", 0, 1))
        #     print("Rank {} sent data {}".format(self.rank, "hello"))
        # elif self.rank == 1:
        #     time.sleep(3)
        #     data = ray.get(self.MPIRuntime.recv.remote(0, 1))
        #     print("Rank {} is received data {}".format(self.rank, data))




# Create two actors
RayMPIRuntime = ray_mpi.RayMPIRuntime.remote()
A = Worker.remote(RayMPIRuntime)
B = Worker.remote(RayMPIRuntime)
C = Worker.remote(RayMPIRuntime)

ray.get(RayMPIRuntime.init.remote([A, B, C], [0, 1, 2]))
ray.get([A.main.remote(), B.main.remote(), C.main.remote()])

