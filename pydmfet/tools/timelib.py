import time

def time0():

    t0 = (time.process_time(), time.perf_counter())

    return t0

def timer(name, t0):

    t1 = (time.process_time(), time.perf_counter())
    tcpu = t1[0] - t0[0]
    twall = t1[1] - t0[1]

    print ("time of", name, ":", '{:.3f}'.format(tcpu), " (cpu), ", '{:.3f}'.format(twall), " (wall) ")

    return t1
