#1157 단어 공부


#입력 - 알파벳 대소문자로 된 단어, 출력 - 가장 많이 사용된 알파벳 대문자/여러개 존재하면 ?출력



alphabet = "abcdefghijklmnopqrstuvwxyz"
word = input()
word = word.lower()

most_use = ""
count = -1

for i in alphabet:
    count_in = word.count(i.lower())

    if(count==count_in):
        most_use = "?"
    elif(count<count_in):
        most_use = i
        count = count_in

print(most_use.upper())



