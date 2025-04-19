#copy mutable elements, actual copy of object customizing

#shallow copy is only one level deep, only references of nested child objects
# deep copy: full independent copy
import copy



org = 5 # immutable type
cpy = org

print(org, cpy)
cpy = 4
print(org, cpy)

org = [3,5,6,7] # mutable type
cpy = org

print(org, cpy)
cpy[0] = 400
print(org, cpy)


org = [3,5,6,7] # mutable type
cpy = copy.copy(org)

print(org, cpy)
cpy[0] = 400
print(org, cpy)


org = [3,5,6,7] # mutable type
cpy = org.copy()

print(org, cpy)
cpy[0] = 400
print(org, cpy)


org = [3,5,6,7] # mutable type
cpy = list(org)

print(org, cpy)
cpy[0] = 400
print(org, cpy)


org = [3,5,6,7] # mutable type
cpy = org[:]

print(org, cpy)
cpy[0] = 400
print(org, cpy)


org = [[3,5,6,7], [33,55,66,77]] # mutable type
cpy = org[:]

print(org, cpy)
cpy[0][0] = 400
print(org, cpy)


org = [[3,5,6,7], [33,55,66,77]] # mutable type
cpy = copy.deepcopy(org)

print(org, cpy)
cpy[0][0] = 400
print(org, cpy)



class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person('Alex', 22)
p2 = p1
p2.age = 566
print(p1.age, p2.age)

p2 = copy.deepcopy(p1)
p2.age = 566
print(p1.age, p2.age)

p3 = copy.copy(p1)
p3.age = 800
print(p1.age, p2.age, p3.age)




class Company:
    def __init__(self, boss, employee):
        self.boss = boss
        self.employee = employee

p1 = Person('Mohammedhamdan', 43)
p2 = Person('Hani', 22)

company = Company(p1, p2)

company_clone = copy.copy(company)
company_clone.boss.age = 788
print(company_clone.boss.age)
print(company.boss.age)

company_clone = copy.deepcopy(company)
company_clone.boss.age = 90
print(company_clone.boss.age)
print(company.boss.age)