#11720 숫자의 합

#입력 - 첫째 줄: 숫자개수 N, 숫자N개 공백없이
#N개의 수 합 출력
import sys

N = input()
N= int(N)


X = sys.stdin.readline().strip()

result = 0
for i in range(N):
    result += int(X[i])

print(result)