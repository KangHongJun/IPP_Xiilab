#10951
#try except를 사용하여 프로그램 종료

import sys

while True:
    try:
        N, X = sys.stdin.readline().split()
        print(int(N)+int(X))
    except:
        break




