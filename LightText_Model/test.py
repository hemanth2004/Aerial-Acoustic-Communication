"""
A homogeneous fixed-size queue.
"""

from array import array

class FixedsizeQueue(object):
"""
A fixed size queue is a homogeneous FIFO queue that can't grow.
"""
def __init__(self, max_size, typecode='i'):
self.size = 0
self.head = 0
self.tail = 0
self.typecode = typecode
self.max = max_size
self._data = None
return

@property
def data(self):
"""
        :return: an array of size self.max, type self.typecode
        """
if self._data is None:
self._data = array(self.typecode, [0 for i in range(self.max)])
return self._data

def enqueue(self, item):
"""
        :param:

         - `item`: the item to add to the queue

        :return: True if added, False if full
        """
if self.size == self.max:
return False

self.data[self.tail] = item

self.size += 1
self.tail += 1

if self.tail == self.max:
self.tail = 0
return True

def dequeue(self):
"""
        :return: oldest item or None
        """
if self.size == 0:
return
item = self.data[self.head]

self.size -= 1
self.head += 1

if self.head == self.max:
self.head = 0
return item

def reset(self):
"""
        :postcondition: head, tail and size reset to 0
        """
self.size = 0
self.tail = 0
self.head = 0
return

def empty(self):
"""
        :return: True if the queue is empty.
        """
return self.size == 0

def full(self):
"""
        :return: True if the queue is full.
        """
return self.size == self.max
# end class FixedsizeQueue