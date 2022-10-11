#4673 셀프넘버 - 일부참고 했음


#
#출력 - 10000이하의 셀프 넘버
Nself_number = []

def self_number(n):
    for num in list(str(n)):
        n += int(num)
    return n

for i in range(1,10001):
    Nself_number.append(self_number(i))

for i in range(1,10001):
    if i not in Nself_number:
        print(i)

