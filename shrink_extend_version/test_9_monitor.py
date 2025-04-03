import ray
import ray_mpi

def main():
    MPIRuntime = ray.get_actor("MPIRuntime")
    while True:
        user_input = input("please input an integer (q to quit): ")
        if user_input.lower() in ["q"]:
            break
        
        try:
            target_num = int(user_input)
            ray.get(MPIRuntime.change_ranks_by_monitor.remote(target_num))
        except ValueError:
            print("not an integer")

if __name__ == "__main__":
    ray.init(address="auto", namespace="default")
    main()
