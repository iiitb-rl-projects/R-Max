rl_glue &
pid1=$!
python My_environment.py &
python rmaxAgent3.py &
python My_experiment.py &
while [ ! -s ./results.csv ]; do sleep 1; done
gnuplot LivePlot.gnu


