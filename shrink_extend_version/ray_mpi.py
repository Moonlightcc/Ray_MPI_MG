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

    def clear(self):
        self.ready_count = 0
        self.ready_event.clear()
    
    def target_change(self, new_target):
        if self.ready_count >= new_target:
            self.ready_event.set()
            self.ready_count = 0
            self.ready_event.clear()


@ray.remote(num_cpus=0)
class RayMPIRuntime:
    def __init__(self):
        self.world_size = -1 # Current active world size
        self.controller = None
        
        # For point-to-point communication
        self.send_buffer = {}

        # For collective communication
        self.collective_signals = None
        self.collective_buffer = []

        # For rank reconfiguration
        self.rank_states = [] # each entry is [obj_ref] to prevent de-reference
        self.rank_active = []

        self.reconfig_signal = None
        self.checkpoint_count = 0
        self.should_change_rank = False 
        self.should_recover = False
        self.should_checkpoint = False
        self.target_rank = -1
        
        self.reborn_flag = []
        self.__reconfig_signal_actor = None

        self.RECONFIG_THRESHOLD = 100 # Number of ranks * number of iterations

        # For profile
        self.enable_profile = False

    def init(
        self,
        world_size: int,
        enable_profile = False
    ):
        self.world_size = world_size
        self.controller = ray.get_actor("Controller")

        self.rank_states = [None] * self.world_size
        self.rank_active = [True] * self.world_size

        # Create 2-D array for send-recv
        self.send_buffer = {}

        self.collective_signals = SignalActor()
        self.collective_buffer = [None] * self.world_size

        self.reconfig_signal = SignalActor()
        self.checkpoint_count = 0
        self.should_change_rank = False 
        self.should_recover = False
        self.should_checkpoint = False
        self.target_rank = -1

        self.reborn_flag = [False] * self.world_size
        self.__reconfig_signal_actor = SignalActor()
        
        self.RECONFIG_THRESHOLD = 100 # Number of ranks * number of iterations
        
        # For profile
        self.enable_profile = enable_profile
        if self.enable_profile == True:
            self.initialization_time = [{} for _ in range(self.world_size)]
            self.computation_time = [{} for _ in range(self.world_size)]
            self.reconfig_time = [{} for _ in range(self.world_size)]
            self.current_iteration = 0
    
    # For profile
    async def set_init_start_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.initialization_time):
            print(f"{rank} exceeds the size of initialization time size. Need to reconfig")
        self.initialization_time[rank][self.current_iteration] = [time, None]

    async def set_init_end_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.initialization_time):
            print(f"{rank} exceeds the size of initialization time size. Need to reconfig")
        if self.current_iteration not in self.initialization_time[rank]:
            print(f"fail to set the start initialization time for rank {rank} at iteration {self.current_iteration}")
        self.initialization_time[rank][self.current_iteration][1] = time
    
    async def set_computation_start_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.computation_time):
            print(f"{rank} exceeds the size of computation time size. Need to reconfig")
        self.computation_time[rank][self.current_iteration] = [time, None]

    async def set_computation_end_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.computation_time):
            print(f"{rank} exceeds the size of computation time size. Need to reconfig")
        if self.current_iteration not in self.computation_time[rank]:
            print(f"fail to set the start computation time for rank {rank} at iteration {self.current_iteration}")
        self.computation_time[rank][self.current_iteration][1] = time
    
    async def set_reconfig_start_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.reconfig_time):
            print(f"{rank} exceeds the size of reconfig time size. Need to reconfig")
        self.reconfig_time[rank][self.current_iteration] = [time, None]

    async def set_reconfig_end_time(self, rank, time):
        if self.enable_profile == False:
            return
        if rank >= len(self.reconfig_time):
            print(f"{rank} exceeds the size of reconfig time size. Need to reconfig")
        if self.current_iteration not in self.reconfig_time[rank]:
            print(f"fail to set the start reconfig time for rank {rank} at iteration {self.current_iteration}")
        self.reconfig_time[rank][self.current_iteration][1] = time
    
    async def profile_output(self, output_file):
        if self.enable_profile == False:
            print("Profile is disabled")
            return
        
        import csv
        def extract_duration(data_list, section_name):
            num_ranks = len(data_list)
            result = {}  # iteration -> {rank: duration}
            for rank in range(num_ranks):
                iter_dict = data_list[rank]
                for iteration, (start, end) in iter_dict.items():
                    if iteration not in result:
                        result[iteration] = {}
                    if end != None:
                        result[iteration][rank] = end - start
            return result
        
        init_data = extract_duration(self.initialization_time, "Initialization")
        comp_data = extract_duration(self.computation_time, "Computation")
        reconf_data = extract_duration(self.reconfig_time, "Reconfig")

        iterations = sorted(set(self.computation_time[0].keys()))
        num_ranks = len(self.computation_time)

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            def write_section(title, data_dict):
                writer.writerow([f"{title} Time (seconds)"])
                header = ["Iteration"] + [f"Rank {r}" for r in range(num_ranks)]
                writer.writerow(header)
                for iteration in iterations:
                    row = [iteration]
                    for rank in range(num_ranks):
                        duration = data_dict.get(iteration, {}).get(rank, "")
                        row.append(duration)
                    writer.writerow(row)
                writer.writerow([])
                
            write_section("Initialization", init_data)
            write_section("Computation", comp_data)
            write_section("Reconfig", reconf_data)

    ########## For rank mellablility ##########
    async def set_threshold(self, threshold):
        self.RECONFIG_THRESHOLD = threshold

    async def reconfigure_test(self, rank):
        # Counter 
        if rank == 0:
            self.checkpoint_count += 1
            if self.checkpoint_count > self.RECONFIG_THRESHOLD:
                self.should_checkpoint = True          
        
        if self.reborn_flag[rank] == True:
            return True

        await self.reconfig_signal.wait(self.world_size)
        
        if rank ==0 and self.enable_profile == True:
            self.current_iteration += 1

        return self.should_change_rank or self.should_recover or self.should_checkpoint

    async def change_ranks(self, rank, num_ranks: int):
        if self.reborn_flag[rank] == True:
            return
        
        if rank == 0 and num_ranks != self.world_size:
            self.should_change_rank = True
            self.target_rank = num_ranks
        
        await self.reconfig_signal.wait(self.world_size)
    
    async def change_ranks_by_monitor(self, num_ranks):
        if num_ranks != self.world_size:
            self.should_change_rank = True
            self.target_rank = num_ranks
    
    async def should_change_rank(self, rank):
        if self.reborn_flag[rank] == True:
            return True
        
        await self.reconfig_signal.wait(self.world_size)

        return self.should_change_rank or self.should_recover or self.should_checkpoint

    async def report_rank_failure(self, failed_rank: int):
        print("MPIRuntime: Rank {} failed".format(failed_rank))

        self.rank_active[failed_rank] = False
        self.world_size -= 1
        self.should_recover = True

        # Skip all the sync so the program could reach recovery process
        self.collective_signals.target_change(self.world_size)
        self.reconfig_signal.target_change(self.world_size)

    async def pre_metadata_update(self, new_num_ranks):
        if new_num_ranks > self.world_size:
            self.rank_states = [None] * new_num_ranks
            self.rank_active = [True] * new_num_ranks
            self.collective_buffer = [None] * new_num_ranks
            self.reborn_flag = [False] * new_num_ranks
            # await self.controller.pre_restrict_rank_handle.remote(new_num_ranks)

        # for profile data, we only want to add new rank but not to delete the old data
        if self.enable_profile == True:
            current_size = len(self.initialization_time)
            if new_num_ranks > current_size:
                self.initialization_time += [{} for _ in range(new_num_ranks-current_size)]
                self.computation_time += [{} for _ in range(new_num_ranks-current_size)]
                self.reconfig_time += [{} for _ in range(new_num_ranks-current_size)]
    
    async def post_metadata_update(self):
        self.rank_states = self.rank_states[:self.world_size]
        self.rank_active = self.rank_active[:self.world_size]
        self.collective_buffer = self.collective_buffer[:self.world_size]
        self.reborn_flag = self.reborn_flag[:self.world_size]
        # await self.controller.post_restrict_rank_handle.remote(new_num_ranks)
    
    async def get_target_num(self):
        return self.target_rank

    async def is_reborn(self, rank):
        return self.reborn_flag[rank]

    async def reconfigure(self, rank: int, state_ref_l, reconfig_handler):
        # TODO: auto-recover might not work well. Add manual recovery process
        if self.should_recover:
             
            # Check current alive ranks

            # Redistribute data from last checkpoint to current alive ranks
            if rank == 0 and self.rank_states[rank] is None:
                print("MPIRuntime: Failure happens and no checkpoint exists. Please restart the whole program")
                return None

            # Change meta data, send_receive buffer / collective buffer

            # Wait until all ranks are ready
            await self.reconfig_signal.wait(self.world_size)
            
            if self.should_change_rank and rank == 0:
                print("MPIRuntime: Rank change to {} ranks fails, as failure happens. Please check current alive ranks and resubmit rank change".format(self.world_size))
            
            self.should_recover = False
            self.should_change_rank = False
            self.should_checkpoint = False
            self.reconfig_test_count = 0

            # Return new state for current rank
            return None

        if self.should_change_rank: # TODO:
            if rank == 0:
                await self.pre_metadata_update(self.target_rank)
            if self.reborn_flag[rank] == False:
                await self.collective_signals.wait(self.world_size)
            new_state = await self.rank_reconfig(rank, state_ref_l, reconfig_handler, self.target_rank)
            if new_state == None:
                return None
            if rank == 0:
                await self.post_metadata_update()
            self.should_change_rank = False
            await self.reconfig_signal.wait(self.target_rank)
            return new_state

        if self.should_checkpoint:
            self.rank_states[rank] = state_ref_l[0]
            if rank == 0:
                self.reconfig_test_count = 0
            await self.collective_signals.wait(self.world_size)
            self.should_checkpoint = False
            self.checkpoint_count = 0
            return state_ref_l[0]

    # only used for in-loop change. The number of actor before and after iteration should be the same
    async def shrink_rank(self, rank: int, state_ref_l, shrink_barrier = False, target_num = -1):
        
        self.rank_active[rank] = False
        self.world_size -= 1
        self.rank_states[rank] = state_ref_l[0]
        ray.get(self.controller.delete_rank.remote(rank))
        if shrink_barrier == True:
            await self.collective_signals.wait(target_num)
    
    # only used for in-loop change.
    async def expand_rank(self, rank: int):
        if self.rank_active[rank]:
            return

        await self.controller.create_and_run_rank.remote(rank, [self.rank_states[rank]])
        self.world_size += 1
        self.rank_active[rank] = True
    
    # only used for in-loop change.
    async def expand_rank_using_new_state(self, rank: int, state_ref_l):
        if self.rank_active[rank]:
            return

        self.rank_states[rank] = state_ref_l[0]
        await self.controller.create_and_run_rank.remote(rank, [self.rank_states[rank]])
        self.world_size += 1
        self.rank_active[rank] = True

    async def rank_reconfig_with_list(self, rank, state_ref_l, reconfig_handler, source_rank_list, target_rank_list, control_rank=-1):
        world_size = len(source_rank_list)
        new_world_size = len(target_rank_list)
        total_survive_process_count = self.world_size + (new_world_size - world_size)
        if rank not in source_rank_list and rank not in target_rank_list:
            await self.__reconfig_signal_actor.wait(total_survive_process_count)
            return state_ref_l[0] # nothing to do
        
        if rank in source_rank_list and self.reborn_flag[rank] == False:
            self.rank_states[rank] = state_ref_l[0]
            await self.collective_signals.wait(world_size)
            
        if control_rank < 0:
            control_rank = source_rank_list[0]
        
        if rank in source_rank_list and self.reborn_flag[rank] == False:
            if rank == control_rank:
                source_state_list = [self.rank_states[i] for i in source_rank_list]
                redistributed_state_list = reconfig_handler(source_state_list, world_size, new_world_size)
                for i, idx in enumerate(target_rank_list):
                    self.rank_states[idx] = redistributed_state_list[i]
                await self.collective_signals.wait(world_size)
            else:
                await self.collective_signals.wait(world_size)
        reborn_rank_list = [rank for rank in target_rank_list if rank not in source_rank_list]
        dead_rank_list = [rank for rank in source_rank_list if rank not in target_rank_list]
        if self.reborn_flag[rank] == False:
            # step 1: activate the actor in target but not source
            new_actor_nums = len(reborn_rank_list)
            current_upper = 0
            current_source_index = source_rank_list.index(rank)
            while new_actor_nums > 0:
                round_new_actor_num = min(new_actor_nums, world_size)
                if current_source_index < round_new_actor_num:
                    dest_rank = reborn_rank_list[current_upper + current_source_index]
                    self.reborn_flag[dest_rank] = True
                    target_state = self.rank_states[dest_rank]
                    await self.controller.create_and_run_rank.remote(dest_rank, [self.rank_states[dest_rank]])
                new_actor_nums -= round_new_actor_num
                current_upper += round_new_actor_num
            # step 2: deactivate the actor in source but not in target
            if rank in dead_rank_list:
                self.rank_active[rank] = False
                ray.get(self.controller.delete_rank.remote(rank))
                return None
            await self.collective_signals.wait(new_world_size)
        else:
            self.reborn_flag[rank] = False
            await self.collective_signals.wait(new_world_size)
        self.world_size = total_survive_process_count
        self.rank_active[rank] = True
        await self.__reconfig_signal_actor.wait(total_survive_process_count)
        return self.rank_states[rank]
    
    async def rank_reconfig(self, rank, state_ref_l, reconfig_handler, new_world_size):
        source_rank_list = list(range(self.world_size))
        target_rank_list = list(range(new_world_size))
        return await self.rank_reconfig_with_list(rank, state_ref_l, reconfig_handler, source_rank_list, target_rank_list)
    
       ########## Communications ##########

    async def get_size(self):
        return self.world_size
    
    async def send(self, data, src_rank: int, dest_rank: int):
        """
        Sends data from src_rank to dest_rank with synchronization to avoid overwriting.
        """
        # Expand the rank that is used, if it is not active
        if not self.rank_active[dest_rank]:
            self.expand_rank(dest_rank)
        if not self.rank_active[src_rank]:
            self.expand_rank(src_rank)

        if (src_rank, dest_rank) not in self.send_buffer:
            self.send_buffer[(src_rank, dest_rank)] = [SignalActor(), data]
        else:
            self.send_buffer[(src_rank, dest_rank)][1] = data

        # Convention: use the lock of the source rank
        # Notify the destination that the data is ready using an event signal
        await self.send_buffer[(src_rank, dest_rank)][0].wait(2)

        # Wait for the receiver to acknowledge that the data has been received
        await self.send_buffer[(src_rank, dest_rank)][0].wait(2)  # Acknowledge signal

    async def recv(self, src_rank: int, dest_rank: int):
        """
        Receives data from src_rank to dest_rank with acknowledgment to avoid overwriting.
        """
        # Expand the rank that is used, if it is not active
        if not self.rank_active[dest_rank]:
            self.expand_rank(dest_rank)
        if not self.rank_active[src_rank]:
            self.expand_rank(src_rank)

        if (src_rank, dest_rank) not in self.send_buffer:
            self.send_buffer[(src_rank, dest_rank)] = [SignalActor(), None]

        # Wait for the sender to notify that the data is ready
        await self.send_buffer[(src_rank, dest_rank)][0].wait(2)

        # Receive the data
        data = self.send_buffer[(src_rank, dest_rank)][1]

        # Clear the buffer after receiving the data
        self.send_buffer[(src_rank, dest_rank)][1] = None

        # Acknowledge that the data was received so the sender can proceed with the next send
        await self.send_buffer[(src_rank, dest_rank)][0].wait(2)

        return data
    
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
   
    async def get_world_size(self):
        return self.world_size
    
    async def barrier_with_target_num(self, rank, target_num):
        await self.collective_signals.wait(target_num)

    async def barrier(self, rank):
        """
        Synchronizes all processes to ensure that all reach this point before proceeding.
        """

        # print(f"Rank {my_rank} is waiting at the barrier with {self.world_size} processes.")

        # Each process waits for all other processes to reach the barrier
        await self.collective_signals.wait(self.world_size)
