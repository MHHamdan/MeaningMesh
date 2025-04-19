#lambda arguments: experession

#map (fun, seq)
add10 = lambda x: x + 10

print(add10(5))

def add_10_fun(x):
    return x + 10

print(add_10_fun(5))

mul = lambda x,y: x * y

print(mul(2,4))

points2D = [(1, 2), (15,1), (5,-1), (10,4)]

points2D_sorted = sorted(points2D)

print(points2D)
print(points2D_sorted)

points2D_sorted = sorted(points2D, key=lambda x:x[1])

print(points2D)
print(points2D_sorted)


def sort_by_y(x):
    return x[1]

points2D_sorted = sorted(points2D, key=sort_by_y)

print(points2D)
print(points2D_sorted)


points2D_sorted = sorted(points2D, key=lambda x: x[0] + x[1])

print(points2D)
print(points2D_sorted)


#

a = [1,2,3,4,5,6,7]
b = map(lambda x:x*2, a)
print(a)
print(list(b))

c = [x * 2 for x in a]
print(c)

#filter (fun, seq)

b = filter(lambda x: x%2 ==0, a)
print(list(b))

c = [x for x in a if x % 2 == 0]
print(c)


#reduce (fun, seq)
from functools import reduce

prod_a = reduce(lambda x,y: x*y , a)
print(prod_a)
