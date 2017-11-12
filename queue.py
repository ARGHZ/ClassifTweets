# -*- coding: utf-8 -*-
from threading import Thread, Event
from Queue import Queue
import time
import random

__author__ = 'Juan David Carrillo López'


class Producer(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        for i in range(10):
            item = random.randint(0, 256)
            self.queue.put(item)
            print 'Procucer notify: item No {0} appended to queue by {1} \n'.format(item, self.name)

class Consumer(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            item = self.queue.get()
            print 'Consumer notify: {0} popped from queue by {1}'.format(item, self.name)
            time.sleep(1)
            self.queue.task_done()

if __name__ == '__main__':
    queue = Queue()
    t1 = Producer(queue)
    t2 = Consumer(queue)
    t3 = Consumer(queue)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()