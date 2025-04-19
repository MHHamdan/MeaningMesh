# from multiprocessing import Process, Value, Array, Lock
# import os
# import time
# from queue import Queue
#
#
# def sequare_number():
#     for i in range(1000):
#         r = i *  i
#
# def add_100(num, lock):
#     for i in range(100):
#         time.sleep(0.01)
#         with lock:
#             num.value += 1
#
# def add_100_arr(nums, lock):
#     for i in range(100):
#         time.sleep(0.01)
#
#         for i in range(len(nums)):
#             with lock:
#                 nums[i] +=  1
#
#
# print('shared arry .... ')
#
# if __name__ == "__main1__":
#
#     lock = Lock()
#     shared_numer = Value('i', 0)
#     print("Number at beginning is ", shared_numer.value)
#
#     p1 = Process(target=add_100, args=(shared_numer,lock))
#     p2 = Process(target=add_100, args=(shared_numer,lock))
#     p3 = Process(target=add_100, args=(shared_numer,lock))
#
#     p1.start()
#     p2.start()
#     p3.start()
#
#     p1.join()
#     p2.join()
#     p3.join()
#
#
#     print("Number at end is ", shared_numer.value)
#
#
#
# if __name__ =="__main2__":
#     lock = Lock()
#     shared_arry = Array('d', [0.0, 100.0, 200.0])
#     print("Array at beginning is ", shared_arry[:])
#
#     p1 = Process(target=add_100_arr, args=(shared_arry, lock))
#     p2 = Process(target=add_100_arr, args=(shared_arry, lock))
#     p3 = Process(target=add_100_arr, args=(shared_arry, lock))
#
#     p1.start()
#     p2.start()
#     p3.start()
#
#     p1.join()
#     p2.join()
#     p3.join()
#
#     print("Array at end is ", shared_arry[:])
#
#
# def square(numbers, queue):
#     for i in numbers:
#         queue.put(i * i)
#
#
# def make_negative(numbers, queue):
#     for i in numbers:
#         queue.put(i * -1)
#
#
# if __name__ == "__main__":
#
#     numbers = range(1, 6)
#     q = Queue()
#
#     p1 = Process(target=square, args=(numbers, q))
#     p2 = Process(target=make_negative, args=(numbers, q))
#
#     p1.start()
#     p2.start()
#
#     p1.join()
#     p2.join()
#
#     # order might not be sequential
#     while not q.empty():
#         print(q.get())
#
#     print('end main')


# communicate between processes with the multiprocessing Queue
# Queues are thread and process safe
from multiprocessing import Process, Queue, Pool


def square(numbers, queue):
    for i in numbers:
        queue.put(i * i)


def make_negative(numbers, queue):
    for i in numbers:
        queue.put(i * -1)


if __name__ == "__main3__":

    numbers = range(1, 6)
    q = Queue()

    p1 = Process(target=square, args=(numbers, q))
    p2 = Process(target=make_negative, args=(numbers, q))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # order might not be sequential
    while not q.empty():
        print(q.get())

    print('end main')


def cube(n):
    return n * n * n

if __name__ == "__main__":

    nums = range(10)
    pool = Pool()




    # map, apply, join, close,
    result = pool.map(cube, nums)

    #r= pool.apply(cube, nums[0])


    pool.close()
    pool.join()

    print(result)


