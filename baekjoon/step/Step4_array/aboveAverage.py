#4344 평균은 넘겠지

#input - 1.개수 C, 2.~  학생수N 점수 점수...
#output - 줄마다 평균 넘는 학생의 비율


#각 줄마다 평균을 구하고, 평균을 넘는 수를 구하여 비율 계산



import sys
N = sys.stdin.readline().strip()
N = int(N)
above_per = []

for i in range(N):
    score_list = []
    average = 0
    above_count = 0

    #각 평균
    score_list.extend(map(int, input().split()))
    for score in range(len(score_list)-1):
        average += score_list[score+1]
    average = average/score_list[0]
    #above 평균
    #반복을 score_list값으로 돌려서 반례 2 1 2에 에러경험
    for score in range(len(score_list)-1):
        if(score_list[score+1]>average):
            above_count +=1
    above_per.append(round(above_count/score_list[0]*100,3))

for per in above_per:
    print("{:.3f}%".format(per))
