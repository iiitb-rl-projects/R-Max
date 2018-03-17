plot "results.csv" using 1:2 title "Avg Reward" with lines,\
"results.csv" using 1:3 title "Std. Dev." with lines
pause 1
reread
