#15596 정수 N개의 합

#함수작성
#매개변수 - a: 합을 구해야하는 정수 n개가 저장되어 있는 리스트
#리턴값 - a에 포함된 정수n개의 합

a = [1,2,3]


def solve(a):
    sum=0
    for i in a:
        sum += i
    return sum



print(solve(a))