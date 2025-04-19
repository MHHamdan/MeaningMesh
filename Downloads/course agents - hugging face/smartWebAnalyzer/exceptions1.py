#Errors and Exceptions

#f = open('fs.txt')


a = [1,2,3]
a.remove(1)
print(a)

#a.remove(1)
#print(a)

mydict = {"name": "Mohammed"}
#print(mydict['age'])

x = 15

if x < 0:
    raise Exception('x should be positive')


x = 3

assert (x >= 0), 'x is not positive'

try:
    a = 5 / 0
except:
    print('an error happens')
print('after')

try:
    a = 5 / 0
except Exception as e:
    print(e)
print('after error --- ')


try:
    a = 5 / '0'
    b = a * 4

except ZeroDivisionError as e:
    print(e)
except TypeError as e:
    print(e)
else:
    print('Everythin is works fine')
finally:
    print("clean up.. ")

class ValueTooHighError(Exception):
    pass

class ValueTooSmallError(Exception):
    def __init__(self, message, value):
        self.message = message
        self.value = value


def test_value(x):
    if x > 100:
        raise ValueTooHighError('Value is too high')
    if x < 5:
        raise ValueTooSmallError("Too small", x)

print(test_value(555))

try:
    test_value(4)
except ValueTooHighError as e:
    print(e)
except ValueTooSmallError as e:
    print(e.message, e.value)
