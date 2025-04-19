#context mangers: resources mangaments, with open statement, with locks


#with open statement much cleaner and much concise
with open('notes.txt', 'w') as file:
    file.write('Some to do ... ')






file = open('notes.txt','w')
try:
    file.write('me me me try to write .... ')
finally:
    file.close()

with open('notes.txt', 'w') as file:
    file.write('Some to do ... ')



#lock



from threading import Lock

lock = Lock()


lock.acquire()
#................................................................
lock.release()

#better way using with locks
with lock:
    #do process here to
    pass



class ManagedFile:
    def __init__(self, filename):
        print('init')
        self.filename = filename

    def __enter__(self):
        print('enter ..')
        self.file = open(self.filename,'w')
        return self.file


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.file:
            self.file.close()
        if exc_type is not None:
            print('exception has been handled ..')
        #print('exc ', exc_type, exc_value)
        print('exit')
        return True



with ManagedFile('notes.txt') as file:
    print('Do some stuff ......')
    file.write('me from my class mangaments resouces .... do some thing')
    file.somemethod()



print('continuing ..')



print('from contextlib import contextmanager' * 4)
from contextlib import contextmanager


@contextmanager
def open_mangaged_file(filename):
    f = open(filename, 'w')
    try:
        yield f
    finally:
        f.close()


with open_mangaged_file('notes.txt') as f:
    f.write('Some text some from decorated and geneated file is ')