# *******
#### asterisks for mul, createion list, tuples, arks, keword, unpakings containers

result = 5 * 5
print(result)

re = 3 ** 3
print(re)

zeros = [0] * 10
print(zeros)

zeros_ones = [0, 1] * 10
print(zeros_ones)

zeros_ones_tuples = (0, 1) * 10
print(zeros_ones_tuples)

string = 'ABC'* 10
print(string)


def foo(a, b, *args, **kwargs):
    print(a, b)
    for arg in args:
        print(arg, end=' ')
    for key in kwargs:
        print('\n', key, kwargs[key], end=' ')


foo(1,10, 44, 99, 2, 3, 4, 5, six=6, seven=7, eight=8)



def foo(a, b, *, c):
    print('\n', a, b, c)


foo(1, 5,c=3)


def foolist(a,b,c):
    print(a,b,c)

mylist1 = (10,30,40)

foolist(*mylist1)

mylist = [10,30,40]

foolist(*mylist)

mydict = {'a':800, 'b':500, 'c':400}
foolist(**mydict)


numbers = [1,2,3,4,5,6,7]

*beginning, last = numbers
print(beginning)
print(last)

numberstuple = (1,2,3,4,5,6,7)

*beginning, last = numberstuple
print(beginning)
print(last)


numbers = [1,2,3,4,5,6,7]

beginning, *last = numbers
print(beginning)
print(last)

numberstuple = (1,2,3,4,5,6,7)

beginning, *last = numberstuple
print(beginning)
print(last)

numbers = [1,2,3,4,5,6,7]

beginning, *middel, last = numbers
print(beginning)
print(middel)
print(last)

numberstuple = (1,2,3,4,5,6,7)

beginning, *middel, last = numberstuple
print(beginning)
print(middel)
print(last)


numberstuple = (1,2,3,4,5,6,7)

beginning, *middel,secondlast, last = numberstuple
print(beginning)
print(middel)
print(secondlast)
print(last)


my_tuple = (1,2,3)
my_list = [4,5,6]

new_list = [*my_tuple, *my_list]
print(new_list)

my_tuple = (1,2,3)
my_set = {4,5,6}

new_list = [*my_tuple, *my_set]
print(new_list)

dict_a = {'a': 1,'b':2, 'c':3, 'd':4}
dict_b = {'e': 5,'f':6, 'g':7, 'h':8}

my_dict = {**dict_a, **dict_b}