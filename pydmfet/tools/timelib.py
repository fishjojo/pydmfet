import time

def timer(name, t0):

    t1 = (time.clock(), time.time())
    tcpu = t1[0] - t0[0]
    twall = t1[1] - t0[1]

    print "time of", name, ":", '{:.3f}'.format(tcpu), " (cpu), ", '{:.3f}'.format(twall), " (wall) "

    return t1
