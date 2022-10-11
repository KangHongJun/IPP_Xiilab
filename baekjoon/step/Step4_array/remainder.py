#3052 나머지

#10가지 숫자 하나씩 입력 0< x <1000
#나머지가 다른 값만큼 list(set(배열))사용하여 중복 제거

List = []

import sys
for i in range(10):
    a = sys.stdin.readline().strip()
    a=int(a)
    List.append(a%42)

result = list(set(List))

print(len(result))
