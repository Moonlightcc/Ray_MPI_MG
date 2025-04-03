import ray
import ray_mpi
import time
import asyncio
from ray.autoscaler._private.cli_logger import cli_logger

ray.init(address="auto", namespace="default")

# Controller actor that manage the lifetime of the ranks
@ray.remote
class Controller:
    def __init__(self, world_size):
        self.world_size = world_size
        self.ranks_handle = {}
        self.finished_count = 0
        self.ranks_condition = asyncio.Condition() # use for tracking finish

    async def create_and_run_rank(self, rank, state_ref_l):
        if rank in self.ranks_handle:
            return 

        rank_handle = Worker.options(name="rank-{}".format(rank), lifetime="detached", get_if_exists=True).remote(rank, state_ref_l)
        self.ranks_handle[rank] = rank_handle

        asyncio.create_task(self._run_rank(rank))

    async def _run_rank(self, rank):
        try:
            await self.ranks_handle[rank].main.remote()
            async with self.ranks_condition:
                self.finished_count += 1
                self.ranks_condition.notify_all()

        except Exception as e:
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
        else:
            self.state = ray.get(state_ref_l[0])
            self.phase = self.state["phase"]
    
    def get_created_at(self):
        return self.created_at
        
    def shrink_rank(self):
        state_ref = ray.put(self.state, _owner=self.MPIRuntime)
        ray.get((self.MPIRuntime.shrink_rank.remote(self.rank, [state_ref])))
        ray.actor.exit_actor()
    
    def reconfig_handler(self, source_rank, N, M):
        state_exp_ref = source_rank[0]
        return [state_exp_ref] * M
    
    def update_from_state(self):
        self.phase = self.state["phase"]
    
    def update_state(self):
        self.state["phase"] = self.phase

    def exit(self):
        ray.actor.exit_actor()

    def main(self):
        if self.phase == 0:
            print(f"My rank is {self.rank} at phase {self.phase}")
            self.update_state()
            state_ref = ray.put(self.state, _owner=self.MPIRuntime)
            state_ref = ray.get((self.MPIRuntime.rank_reconfig.remote(self.rank, [state_ref], self.reconfig_handler, 8)))
            new_state = ray.get(state_ref)
            self.state = new_state
            self.update_from_state()
            self.phase += 1
        
        if self.phase == 1:
            print("My rank is {} at phase 1".format(self.rank))
            self.update_state()
            state_ref = ray.put(self.state, _owner=self.MPIRuntime)
            state_ref = ray.get((self.MPIRuntime.rank_reconfig.remote(self.rank, [state_ref], self.reconfig_handler, 16)))
            new_state = ray.get(state_ref)
            self.state = new_state
            self.update_from_state()
            self.phase += 1

        if self.phase == 2:
            print("My rank is {} at phase 2".format(self.rank))


world_size = 16
controller = Controller.options(name="Controller", lifetime="detached", get_if_exists=True).remote(world_size)
RayMPIRuntime = ray_mpi.RayMPIRuntime.options(name="MPIRuntime", lifetime="detached", get_if_exists=True).remote()

ray.get(RayMPIRuntime.init.remote(world_size))
ray.get([controller.create_and_run_rank.remote(i, None) for i in range(world_size)])
ray.get(controller.wait_all_ranks.remote())
ray.get(controller.delete_all_ranks.remote())

ray.kill(RayMPIRuntime)
ray.kill(controller)
