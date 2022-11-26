from mpi4py import MPI
from traprule import Trap

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

a = 0.0
b = 1.0
n = 1024
h = (b - a) / n

dest = 0
total = -1.0
integral = Trap(a, b, n, h)

# Add up the integrals calculated by each process
if my_rank == 0:
    total = integral

    for source in range(1, p):
        integral = comm.recv(source=source)
        print("PE ", my_rank, "<-", source, ",", integral)
        total = total + integral

else:
    print("PE", my_rank, "->", dest, ",", integral)
    comm.send(integral, dest=0)

# Print the result
if (my_rank == 0):
    print("With n=", n, ", trapezoids, \n")
    print("integral from", a, "to", b, "=", total, "\n")

MPI.Finalize

# mpiexec -np <num_of_cores> python trapezoid.py