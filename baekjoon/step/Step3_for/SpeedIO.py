#15552번 속도개선

import sys
A = sys.stdin.readline()
A=int(A)

for i in range(A):
    X, Y = sys.stdin.readline().split()
    print(int(X)+int(Y))


