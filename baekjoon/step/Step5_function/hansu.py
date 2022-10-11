#1065 한수

#양의 정수 X의 각 자리가 등차수열을 이룬다면, 그 수를 한수라고 함
#N이 주어졌을때 1이상,N이하의 한수의 개수 출력

#입력 - 1000이하 자연수 N
#출력 - 1보다 이상 N이하의 한수

#N만큼 반복 99까지는 ++
#100부터는 체크, 공차가 0이어도 성립하므로 결국 등차를 구해서 비교

def arithmetical(N):
    ari_count = 0 #한수

    for i in range(1,N+1):
        ari_list = []
        ari = list(str(i))
        if(i<100):
            ari_count +=1
        else:
            # 등차 저장
            for check in range(len(ari) - 1):
                number = int(ari[check])
                number2 = int(ari[check + 1])

                ari_num = number2 - number
                ari_list.append(ari_num)
            #등차 비교
            if(ari_list[0]==ari_list[1]):
                ari_count += 1

    return ari_count


N = input()

print(arithmetical(int(N)))
