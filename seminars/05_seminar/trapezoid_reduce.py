from mpi4py import MPI
from traprule import Trap
from getdata import Get_data

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

a, b, n = Get_data(my_rank, p, comm) # process 0 will read data

dest = 0
total = -1.0

h = (b - a) / n # h is the same for all processes
local_n = n / p # so is the number of trapezoids
local_a = a + my_rank * local_n * h
local_b = local_a + local_n * h
integral = Trap(local_a, local_b, local_n, h)

# Add up the integrals calculated by each process

total = comm.reduce(integral)

# Print the result
if my_rank == 0:
    print("With n=", n, ", trapezoids, ")
    print("integral from", a, "to", b, "=", total)

MPI.Finalize

# mpiexec -np <num_of_cores> python trapezoid_reduce.py