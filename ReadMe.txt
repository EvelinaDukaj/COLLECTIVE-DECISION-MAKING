________________________________________________________________________________________________________________________________________
Artificial decision making of autonomous vehicles in roundabout entering situations with information exchange-based learning approaches
----------------------------------------------------------------------------------------------------------------------------------------


The project consists of simulated self-driven cars facing roundabout entering situations. Here, the cars must learn how to enter the roundabout without crashing with the other vehicles that are already circulating inside the roundabout.

Inside each roundabout (2) there are obstacle vehicles that only circulate in there and are defined with a constant speed. This is done so that they emulate the dynamic situation inside a real roundabout.

The number of obstacle vehicles, as well as the number of simulated self driven vehicles is defined in “variables.py", here other variables can be defined as well, among them; vehicles speed, path where to save the training and testing results, number of training and testing iterations.

In general terms, the project works as follows:

Training:
The vehicles enter the simulation area without any knowledge. Once they reach the entrance of the roundabout they reduce their speed to almost zero and visualize the vehicles inside the roundabout as well as the possible entering positions. Each of these positions is considered as a solution, which are evaluated in the featured space risk vs. waiting time, this way the minimal set is found applying Pareto domination among the solutions.

Once the minimal set is found, a Hypervolume is calculated for this set. this value of hypervolume, the risk of entering in the first position (First_Risk) and several other features, become the rule learnt by the vehicle in the considered roundabout entrance. Each time a vehicle enters the roundabout, the features of new rule is calculated, and depending on if the rule is equal (very similar) to another rule previously learnt or not, once the vehicle reaches the exit of the roundabout the rule is stored as a new rule or as a replacement of an existing one inside the rulebook.

The rulebook is decentralized; so that each vehicle has its own. There is an exchange area defined inside the method "index_for_exchange()" in which the vehicles are allowed to exchange information (rules), and each time 2 vehicles are selected to exchange depending on their provability of selection defined in "rouletteWheel.py". The way in which the vehicles exchange information is defined by the approach:

Best & Worst (B&W): Exchange of information depending on the fitness of each rule;
1. Filter the rulebook by vehicle's ID "agent_Dictionary()"
2. Determine the rule with the best fitness of one of the vehicles "find_best_fitness()"
3. Determine the rule with the worst fitness of the other vehicle "find_worst_fitness()"
4. Replace the rule of worst fitness with the rule of best fitness "switch_values()"

Cluster of knowledge (COK): Exchange of information in terms of populating with rules the entire featured space (Hypervolume vs First_Risk);
On this approach 2 rules are exchanged each time 2 vehicles are selected for exchange of information; one by center based COK, and one by gap based COK.
Center based COK:
1. Filter the rulebook by vehicle's ID "statesById2()"
2. Form a cluster with the rules of a vehicle and find the center "findCenter()"
3. Find the rule of the other vehicle that is most distant from the center found "findFardest()"
4. Find the closest 2 rules inside the cluster and randomly selects one "findClosest()"
5. Replace the rule randomly selected with the rule found in step 3 "stateExchange()"

Gap based COk:
1. Filter the rulebook by vehicle's ID "statesById2()"
2. Find the 2 rules in the cluster with the biggest empty space between them "findBiggestGap()"
3. Find the middle point of the selected 2 rules and identifies the closest rule from the other vehicle to this point "fillGap()"
4. Find the closest 2 rules inside the cluster and randomly selects one "findClosest()"
5. Replace the rule randomly selected with the rule found in step 3 "stateExchange()"

Random exchange (RAND):
1. Select 2 vehicles randomly
2. Select randomly one rule on each vehicle
3. Add the selected rules to the rulebook of the opposite vehicle
4. Apply crowding distance algorithm to reduce the number of rules in the final rulebook by a user specified percentage

Testing:
The rules learnt by each vehicle during the training phase are evaluated in the simulation, using the most similar rule to the current roundabout entering situation to decide when to enter, and in terms of the result the fitness is calculated.
__________________________*
Running the project:
--------------------------*
1. Go to folder "Tests" in cdm_project/highwayEnv/Tests/
2. If not there already, create the folders: "Archives", "TrainingStatistics", "TestStatistics"
3. Go to the file "variables.py" and define the number of vehicles, training entries, testing entries and path in your computer to the folder "Tests"
4. Go to the file "cmd_project.py" and use the method "approachActivation()" to select the information exchange approach by setting its value to "True" (Set only one approach as True at the time).
5. Use the Seed list and parameter list on this file to determine the parameter/seed combinations in which you want to run the simulation.
6. Define the number of processors you want to use for the simulation "Pool()"
7. Run the simulation, results will be displayed in the folders previously created.
8. To convert the ".pkl" files into ".csv" files the file "readPkl.py" can be used specifying the ".pkl" file name

Note: To visualize the simulated vehicles driving around the simulation path, go to "roundabout.py" and in line 112, change the parameter "offscreen" to "False"


Enjoy the Project!!!

Authors:
______________________*
Supervisor: Sanaz Mostaghim

Team: 3 musketeers
Evelina Dukaj
Ronald Mendez Lara
Mohammed Nadeem
----------------------*