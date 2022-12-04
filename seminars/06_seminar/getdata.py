def Get_data(my_rank, p , comm):
    a = None
    b = None
    n = None

    if my_rank == 0:
        print("Rank ", my_rank, ": Enter a, b, and n")
        a = float(input("enter a: "))
        b = float(input("enter b: "))
        n = int(input("enter n: "))
        print("ready for broadcast")
    
    a = comm.bcast(a)
    b = comm.bcast(b)
    n = comm.bcast(n)

    return a, b, n