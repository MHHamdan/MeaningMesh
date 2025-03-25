#arguments, parameters,positional_arguments, arguments, variables-length,(*args, **kwargs) container, unpacking , local, parameters passing , by refrence ,



def print_name(name):
    print(name)

print_name('Mohammedhamdan')


def foo(a,b,c, d=4):
    print(a,b,c, d)


foo(1,2,3)

foo(c=1,a=2,b=3)

#positional vs keyword arguments
foo(1,c=2,b=3)

foo(1, c=33, b=55)

#variables length arguments

def fooar(*args, **kwargs):
    #* you can pass any number of positional arguments, ** pass any number of keyword arguments
    #print(a)
    for arg in args:
        print(arg, end=' ')

    for key in kwargs:
        print(key, kwargs[key], end=' ')

fooar(1, 2, 3,4,5,  a=44, b=66, c=77)

def lastfoo(*args,last):
    for arg in args:
        print(arg, end=' ')

    print(last)


lastfoo(1, 2,3, last=33)

def fun(a,b,c):
    print(a,b,c)

mylist = [1,3,4]
mytuple = (1,2,3)

fun(*mylist)
fun(*mytuple)


mydict = {'a':1, 'b':2, 'c':3}
fun(**mydict)


number = 0

def localGlobal():
    global number
    x = number
    number = 2
    print(f"Numnber inside function {x}")




localGlobal()
print(number)

print('call by value or call by refrence >>> call by object or call by object refrence')



def foox(x):
    x = 5
#immutabe object cannot be change
var = 10
foox(var)
print(var)

def fool(a):
    a.append(900)
    a[0] = -100
#mutabel object can be change
a = [1,3,4]
fool(a)
print(a)


def fool1(a):
    a = [200, 300, 400]
    a.append(900)
    a[0] = -100
    print(a)
#mutabel object can be change
a = [1,3,4]
fool1(a)
print(a)

def fool1(a):
    a += [200, 300, 400]
    a.append(900)
    a[0] = -100
    print(a)
#mutabel object can be change
a = [1,3,4]
fool1(a)
print(a)


