#2675 문자열 반복

#문자열(QR Code)을 입력받으면 문자마다 반복
#QR Code "alphanumeric" 문자는 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\$%*+-./:
#입력 - 문자열 S, 각 문자 R번 반복, 출력 - 새 문자열 P

T = input()
T = int(T)

repeat_out = []

for i in range(T):
    repeat_in = input().split(" ")

    result_string = ""

    R = int(repeat_in[0])
    S = repeat_in[1]

    for i in S:
        result_string = result_string + i*R
        #print(result_string)
    repeat_out.append(result_string)

for i in repeat_out:
    print(i)