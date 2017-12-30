import os
import multiprocessing
import time

def myfunc(q,i):
    print "child: " + os.environ['FOO']
    os.environ['FOO'] = "child_set_"+str(i)
    print "child new1: " + os.environ['FOO']
    time.sleep(1)
    #q.put(None)
    q.get()
    print "child new2: " + os.environ['FOO']


if __name__ == "__main__":
    os.environ['FOO'] = 'parent_set'
    q = multiprocessing.Queue()
    for i in range(3):
        proc = multiprocessing.Process(target=myfunc, args=(q,i,))
        proc.start()

    print "parent: " + os.environ['FOO']
    os.environ['FOO'] = "parent_set_again"
    for i in range(4):
        q.put(None)
    time.sleep(2)
    print "parent new: " + os.environ['FOO']
    print 'blocking'
    print q.get()
    print 'no blocked'