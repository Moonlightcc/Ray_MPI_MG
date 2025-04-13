import ray
import ray_mpi
import time
import asyncio
import numpy as np
import random
from ray.autoscaler._private.cli_logger import cli_logger

import matplotlib.pyplot as plt

ray.init(address="auto", namespace="default")

# Controller actor that manage the lifetime of the ranks
@ray.remote
class Controller:
    def __init__(self, world_size):
        self.world_size = world_size
        self.ranks_handle = {}
        self.finished_count = 0
        self.ranks_condition = asyncio.Condition()

        self.enable_profile = False

    async def setup_profile(self):
        self.enable_profile = True
        self.MPIRuntime = ray.get_actor("MPIRuntime")
        
    async def create_and_run_rank(self, rank, state_ref_l):
        if self.enable_profile == True:
            current_time = time.time()
            ray.get(self.MPIRuntime.set_init_start_time.remote(rank, current_time))

        if rank in self.ranks_handle:
            return 
        
        rank_handle = Worker.options(name="rank-{}".format(rank), lifetime="detached", get_if_exists=True).remote(rank, state_ref_l)
        self.ranks_handle[rank] = rank_handle

        asyncio.create_task(self._run_rank(rank))
    
    async def _run_rank(self, rank):
        # print("Try to run rank {}".format(rank))
        try:
            # print("Run rank {}".format(rank))
            await self.ranks_handle[rank].main.remote()

            async with self.ranks_condition:
                self.finished_count += 1
                self.ranks_condition.notify_all()

        except Exception as e:
            # print("Rank {} exited".format(rank))
            self.ranks_handle.pop(rank)
            return
    
    async def wait_all_ranks(self):
        async with self.ranks_condition:
            while True:
                if self.finished_count == len(self.ranks_handle):
                    break
                await self.ranks_condition.wait()

    async def delete_all_ranks(self):
        for rank in self.ranks_handle:
            ray.kill(self.ranks_handle[rank])
        self.ranks_handle = {}

    async def pre_restrict_rank_handle(self, target_num):
        return 

    async def post_restrict_rank_handle(self, target_num):
        self.world_size = target_num

    async def delete_rank(self, rank):
         ray.kill(self.ranks_handle[rank])

@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, rank, state_ref_l):
        self.rank = rank
        self.MPIRuntime = ray.get_actor("MPIRuntime")
        self.state = {}
        if state_ref_l is None:
            self.state = {}
            self.phase = 0
            N = 1024
            self.data = np.arange(rank*N, (rank+1)*N)
            self.world_size = 8
        else:
            self.state = ray.get(state_ref_l[0])
            self.phase = self.state["phase"]
            self.data = self.state["data"].copy()
            self.world_size = self.state["world_size"]
    
    def shrink_rank(self):
        state_ref = ray.put(self.state, _owner=self.MPIRuntime)
        ray.get((self.MPIRuntime.shrink_rank.remote(self.rank, [state_ref])))
        ray.actor.exit_actor()
    
    def reconfig_handler(self, source_states, N, M):
        data = [None] * N
        phase = ray.get(source_states[0])["phase"]
        for i in range(N):
            data[i] = ray.get(source_states[i])["data"]
        total_data = np.concatenate(data)
        split_data = np.array_split(total_data, M)
        return_val = [None] * M
        for i in range(M):
            new_state = {}
            new_state["phase"] = phase
            new_state["world_size"] = M
            new_state["data"] = split_data[i]
            return_val[i] = ray.put(new_state)
        return return_val
    
    def update_from_state(self):
        self.phase = self.state["phase"]
        self.data = self.state["data"].copy()
        self.world_size = self.state["world_size"]
    
    def update_state(self):
        self.state["phase"] = self.phase
        self.state["data"] = self.data
        self.state["world_size"] = self.world_size
    
    def update_data(self):
        # simulate a scan process
        N = self.data.shape[0]
        for i in range(N):
            self.data[i] += random.randint(1, 10)
        
        partial_sum = np.sum(self.data)
        gather_data = (self.rank, partial_sum)
        gather_data_array = ray.get(self.MPIRuntime.gather.remote(self.rank, gather_data, root_rank=0))
        if self.rank == 0:
            sorted_data_array = sorted(gather_data_array, key=lambda x: x[0])
            sorted_data_array = [y for (_,y) in sorted_data_array]
            partial_sum_array = [0] * len(sorted_data_array)
            for i in range(1, len(sorted_data_array)):
                partial_sum_array[i] = partial_sum_array[i-1] + sorted_data_array[i-1]
        else:
            partial_sum_array = None
        prefix_sum = ray.get(self.MPIRuntime.scatter.remote(self.rank, partial_sum_array, root_rank=0))
        prefix_sum = prefix_sum[0]
        scan_data = [0] * N
        scan_data[0] = self.data[0] + prefix_sum
        for i in range(1, N):
            scan_data[i] = self.data[i] + scan_data[i-1]
        for i in range(N):
            self.data[i] = self.data[i] / (scan_data[i])

    def reconfigure_rank(self):
        should_reconfig = ray.get(self.MPIRuntime.reconfigure_test.remote(self.rank))
        if not should_reconfig:
            return
        if self.rank == 0:
            target_rank = ray.get(self.MPIRuntime.get_target_num.remote())
            self.reconfig_iteration[self.phase] = (self.world_size, target_rank)
        
        ray.get(self.MPIRuntime.set_reconfig_start_time.remote(self.rank, time.time()))
        self.update_state()
        state_ref = ray.put(self.state, _owner=self.MPIRuntime)
        state_ref = ray.get(self.MPIRuntime.reconfigure.remote(self.rank, [state_ref], self.reconfig_handler))
        new_state = ray.get(state_ref)
        self.state = new_state
        self.update_from_state()
        ray.get(self.MPIRuntime.set_reconfig_end_time.remote(self.rank, time.time()))
        if self.rank == 0:
            print(f"rechedule happens at {self.phase}")

    def exit(self):
        ray.actor.exit_actor()

    def compute_throughput(self, iteration_completion_times, window_size=50):
        sorted_iters = sorted(iteration_completion_times.items())
        iters = [item[0] for item in sorted_iters]
        durations = [item[1] for item in sorted_iters]

        cumulative_times = []
        total_time = 0
        for dur in durations:
            total_time += dur
            cumulative_times.append(total_time)

        throughput = []
        centers = []
        for i in range(len(iters) - window_size):
            start_time = cumulative_times[i]
            end_time = cumulative_times[i + window_size]
            duration = end_time - start_time
            if duration > 0:
                throughput.append(window_size / duration)
            else:
                throughput.append(0)
            centers.append(iters[i + window_size // 2])
        return centers, throughput
    
    def plot_throughput(self, centers, throughput, filename="throughput_plot.png"):

        plt.figure(figsize=(10, 6))
        plt.plot(centers, throughput, 'b.-', label='Throughput')

        for iter_idx, (a, b) in self.reconfig_iteration.items():
            plt.axvline(x=iter_idx, color='red', linestyle='--', alpha=0.6)
            plt.text(iter_idx, max(throughput) * 0.95, f"{a}→{b}", rotation=90,
                    verticalalignment='top', horizontalalignment='center', fontsize=9, color='red')

        plt.xlabel('Iteration Number')
        plt.ylabel('Throughput (Iterations/s)')
        plt.title('Iteration Throughput Over Time with Actor Reconfigurations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"image saved: {filename}")


    def main(self):
        ray.get(self.MPIRuntime.set_init_end_time.remote(self.rank, time.time()))
        if self.rank == 0:
            self.iteration_time = {}
            self.reconfig_iteration = {}
            ray.get(self.MPIRuntime.set_threshold.remote(1000000))
        is_reborn = ray.get(self.MPIRuntime.is_reborn.remote(self.rank))
        if is_reborn == False:
            ray.get(self.MPIRuntime.barrier.remote(self.rank))
        while self.phase < 10000:
            if self.rank == 0:
                if self.phase % 1000 == 0:
                    print(f"Pass phase: {self.phase}")
                start_time = time.time()
            is_reborn = ray.get(self.MPIRuntime.is_reborn.remote(self.rank))
            if is_reborn == False:
                self.update_data()
            self.reconfigure_rank()
            self.phase += 1
            ray.get(self.MPIRuntime.barrier.remote(self.rank))            
            if self.rank == 0:
                end_time = time.time()
                self.iteration_time[self.phase] = end_time - start_time
 
        
        if self.rank == 0:
            print(self.reconfig_iteration)
            centers, throughput = self.compute_throughput(self.iteration_time, window_size=200)
            self.plot_throughput(centers, throughput, filename="my_throughput.png")


world_size = 8
controller = Controller.options(name="Controller", lifetime="detached", get_if_exists=True).remote(world_size)
RayMPIRuntime = ray_mpi.RayMPIRuntime.options(name="MPIRuntime", lifetime="detached", get_if_exists=True).remote()

enable_profile = False
ray.get(RayMPIRuntime.init.remote(world_size, enable_profile))

# for profile
if enable_profile:
    ray.get(controller.setup_profile.remote())

ray.get([controller.create_and_run_rank.remote(i, None) for i in range(world_size)])
ray.get(controller.wait_all_ranks.remote())
ray.get(controller.delete_all_ranks.remote())

if enable_profile:
    output_file = "profile.csv"
    ray.get(RayMPIRuntime.profile_output.remote(output_file))

ray.kill(RayMPIRuntime)
ray.kill(controller)
