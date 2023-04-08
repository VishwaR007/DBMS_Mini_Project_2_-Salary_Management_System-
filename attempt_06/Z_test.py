# importing datetime module for now()
from datetime import date
import calendar

# using today() to get current time
current_date = date.today()
	
# Printing attributes of now().
print ("The attributes of today() are : ")

print("Value of current_date : ", current_date)	

print ("Year: ", end = "")
print (current_date.year)
	
print ("Month: ", end = "")
print (current_date.month)
	
print ("Day: ", end = "")
print (current_date.day)

dayName = calendar.day_name[current_date.weekday()]
print(dayName)


# count_goodproduct = 0
# count_badproduct = 0
# count_withouthead = 0
# count_withouttail = 0

var_day = current_date.day
var_day_arr = [0]
for day in var_day_arr:
    if var_day != day:
        count_goodproduct = 0
        count_badproduct = 0
        count_withouthead = 0
        count_withouttail = 0
        for i in range(1, var_day+1):
            var_day_arr.append(i)
        print(var_day_arr)
    else:
        count_goodproduct += 1
        count_badproduct += 1
        count_withouthead += 1
        count_withouttail += 1
        per_day = [count_goodproduct, count_badproduct, count_withouthead, count_withouttail]


