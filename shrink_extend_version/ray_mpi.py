import asyncio
import ray
from typing import List

class SignalActor:
    def __init__(self):
        self.ready_event = asyncio.Event()
        self.ready_count = 0

    async def wait(self, target_count):
        # Prevent consecutive call which may overwrite previous data
        while self.ready_event.is_set():
            asyncio.sleep(0.1)
        
        self.ready_count += 1
        if self.ready_count < target_count:
            # print("Waiting")
            await self.ready_event.wait()
            # print("Ready")
        else:
            self.ready_event.set()
            self.ready_count = 0
            self.ready_event.clear()
    
    # def release(self):
    #     self.ready_event.set()
    #     self.ready_count = 0
    #     self.ready_event.clear()

    def clear(self):
        self.ready_count = 0
        self.ready_event.clear()


@ray.remote(num_cpus=0)
class RayMPIRuntime:
    def __init__(self):
        self.world_size = -1 # Current active world size
        self.max_world_size = -1 # Maximum world size
        self.controller = None

        # The object reference for rank states
        self.rank_states = [] # each entry is [obj_ref] to prevent de-reference
        self.rank_active = []
        
        # For point-to-point communication
        self.signals = [] 
        self.send_buffer = []

        # For collective communication
        self.collective_signals = None
        self.collective_buffer = []

        

    def init(
        self,
        world_size: int,
    ):
        self.world_size = world_size
        self.max_world_size = world_size
        self.controller = ray.get_actor("Controller")

        self.rank_states = [None] * self.world_size
        self.rank_active = [True] * self.world_size

        # Create signals for sync
        self.signals = [SignalActor() for _ in range(self.world_size)]

        # Create 2-D array for send-recv
        self.send_buffer = [[[SignalActor(), None] for _ in range(self.world_size)] for _ in range(self.world_size)]

        self.collective_signals = SignalActor()
        self.collective_buffer = [None] * self.world_size

    ########## For rank mellablility ##########
    async def shrink_rank(self, rank: int, state_ref_l):
        
        #print("MPIRuntime Shrinking rank {}".format(rank), flush=True)
        
        self.rank_active[rank] = False
        self.world_size -= 1
        self.rank_states[rank] = state_ref_l[0]

    async def expand_rank(self, rank: int):
        if self.rank_active[rank]:
            return

        #print("MPIRuntime Expanding rank {}".format(rank))

        await self.controller.init_rank.remote(rank, [self.rank_states[rank]])
        self.world_size += 1
        self.rank_active[rank] = True
        
    async def expand_rank_using_new_state(self, rank: int, state_ref_l):
        if self.rank_active[rank]:
            return

        self.rank_states[rank] = state_ref_l[0]
        await self.controller.init_rank.remote(rank, [self.rank_states[rank]])
        self.world_size += 1
        self.rank_active[rank] = True

    ########## Communications ##########
    
    async def send(self, data, src_rank: int, dest_rank: int):
        """
        Sends data from src_rank to dest_rank with synchronization to avoid overwriting.
        """
        # Expand the rank that is used, if it is not active
        if not self.rank_active[dest_rank]:
            self.expand_rank(dest_rank)
        if not self.rank_active[src_rank]:
            self.expand_rank(src_rank)

        # The sender places its data in the send buffer for the destination rank
        self.send_buffer[src_rank][dest_rank][1] = data

        # Convention: use the lock of the source rank
        # Notify the destination that the data is ready using an event signal
        await self.send_buffer[src_rank][dest_rank][0].wait(2)

        # Wait for the receiver to acknowledge that the data has been received
        await self.send_buffer[src_rank][dest_rank][0].wait(2)  # Acknowledge signal

    async def recv(self, src_rank: int, dest_rank: int):
        """
        Receives data from src_rank to dest_rank with acknowledgment to avoid overwriting.
        """
        # Expand the rank that is used, if it is not active
        if not self.rank_active[dest_rank]:
            self.expand_rank(dest_rank)
        if not self.rank_active[src_rank]:
            self.expand_rank(src_rank)

        # Wait for the sender to notify that the data is ready
        await self.send_buffer[src_rank][dest_rank][0].wait(2)

        # Receive the data
        data = self.send_buffer[src_rank][dest_rank][1]

        # Clear the buffer after receiving the data
        self.send_buffer[src_rank][dest_rank][1] = None

        # Acknowledge that the data was received so the sender can proceed with the next send
        await self.send_buffer[src_rank][dest_rank][0].wait(2)

        return data
    
    async def scatter(self, my_rank, data, root_rank: int):
        if my_rank == root_rank:
            chunk_size = len(data) // self.world_size
            for dest_rank in range(self.world_size):
                start_idx = dest_rank * chunk_size
                end_idx = start_idx + chunk_size
                self.collective_buffer[dest_rank] = data[start_idx:end_idx]
            await self.collective_signals.wait(self.world_size)
            scattered_data = self.collective_buffer[my_rank]
            self.collective_buffer[my_rank] = None
        else:
            await self.collective_signals.wait(self.world_size)
            scattered_data = self.collective_buffer[my_rank]
            self.collective_buffer[my_rank] = None

        await self.collective_signals.wait(self.world_size)
        return scattered_data


    async def broadcast(self, my_rank, data, root_rank: int):
        """
        Broadcasts data from the root_rank to all other processes asynchronously.
        Uses two synchronization points to avoid overwriting between consecutive operations.
        """
        if my_rank == root_rank:
            # The root places its data in the buffer for all other ranks
            for dest_rank in range(self.world_size):
                if dest_rank != root_rank:
                    self.collective_buffer[dest_rank] = data

            # First synchronization: Ensure the root has placed its data in the buffer
            await self.collective_signals.wait(self.world_size)

        else:
            # Non-root processes wait for the broadcast data from the root
            await self.collective_signals.wait(self.world_size)
            data = self.collective_buffer[my_rank]  # Read the broadcasted data from the buffer
            self.collective_buffer[my_rank] = None  

        # Second synchronization: Ensure all processes have received the data
        await self.collective_signals.wait(self.world_size)  # Prevents starting new operation early

        return data  # Return the broadcasted data for all ranks


    async def reduce(self, my_rank, local_data, root_rank: int, op=sum):
        """
        Performs a reduction operation across all processes, with the result being stored at the root process.
        Uses two synchronization points to avoid overwriting between consecutive operations.
        """
        # Each process places its local data in the collective buffer for the root rank
        self.collective_buffer[my_rank] = local_data

        # First synchronization: Ensure all processes have placed their data
        await self.collective_signals.wait(self.world_size)

        # Only the root process performs the reduction operation
        if my_rank == root_rank:
            reduced_result = self.collective_buffer[root_rank]  # Start with the root's own data
            for src_rank in range(self.world_size):
                if src_rank != root_rank:
                    reduced_result = op([reduced_result, self.collective_buffer[src_rank]])
        else:
            reduced_result = None

        # Second synchronization: Ensure all processes have finished processing before the next operation
        await self.collective_signals.wait(self.world_size)

        return reduced_result  # Only the root process returns the reduced result, others return None


    async def allgather(self, my_rank, local_data):
        """
        Gathers data from all processes and distributes the full dataset to all processes.
        Ensures two synchronizations to avoid overwriting data between consecutive operations.
        """
        # Each process places its local data in the collective buffer
        self.collective_buffer[my_rank] = local_data

        # First synchronization: Ensure all processes have placed their data
        await self.collective_signals.wait(self.world_size)

        # After all processes have placed their data, gather data from all processes
        gathered_data = [self.collective_buffer[src_rank] for src_rank in range(self.world_size)]

        # Second synchronization: Ensure all processes have completed reading the data before next operation
        await self.collective_signals.wait(self.world_size)
        self.collective_buffer[my_rank] = None

        return gathered_data

    async def gather(self, my_rank, local_data, root_rank: int):
        """
        Gathers data from all processes and distributes the full dataset to all processes.
        Ensures two synchronizations to avoid overwriting data between consecutive operations.
        """

        # Each process places its local data in the collective buffer
        self.collective_buffer[my_rank] = local_data

        # First synchronization: Ensure all processes have placed their data
        await self.collective_signals.wait(self.world_size)

        if my_rank == root_rank:
            # Root process gathers data from all other processes
            gathered_data = [self.collective_buffer[src_rank] for src_rank in range(self.world_size)]
        else:
            gathered_data = None
        
        await self.collective_signals.wait(self.world_size)
        self.collective_buffer[my_rank] = None

        return gathered_data

    async def allreduce(self, my_rank, local_data, op=sum):
        """
        Performs a reduction (sum, max, etc.) across all processes and distributes the result to all.
        Ensures two synchronizations to avoid overwriting data between consecutive operations.
        """
        # Each process places its local data in the collective buffer
        self.collective_buffer[my_rank] = local_data

        # First synchronization: Ensure all processes have placed their data
        await self.collective_signals.wait(self.world_size)

        # Perform the reduction operation after all processes have placed their data
        reduced_result = self.collective_buffer[0]
        for src_rank in range(1, self.world_size):
            reduced_result = op([reduced_result, self.collective_buffer[src_rank]])

        # Distribute the reduced result to all processes (each process gets the same result)
        for dest_rank in range(self.world_size):
            self.collective_buffer[dest_rank] = reduced_result

        # Each process now collects the final reduced result
        final_result = self.collective_buffer[my_rank]

        # Second synchronization: Ensure all processes have completed reading the result before next operation
        await self.collective_signals.wait(self.world_size)  
        self.collective_buffer[my_rank] = None

        return final_result
    
    async def barrier(self, my_rank):
        """
        Synchronizes all processes to ensure that all reach this point before proceeding.
        """
        # Each process waits for all other processes to reach the barrier
        await self.collective_signals.wait(self.world_size)

