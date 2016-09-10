# qlearning-test
A simple Q-learning model applied to the following toy example: optimize the bid price of an Agent at node 1 (n1) in a network with three transmission lines (l1-l3), given the known constant bid prices of generators at nodes 2 and 3.
```
         ____
        | n2 |
        |____|
        /     \
    l1 /       \ l3
  ____/_________\____
 | n1 |   l2    | n3 |
 |____|         |____|
```
Uses CVXPY (http://www.cvxpy.org/en/latest/) and Gurobi (http://www.gurobi.com/) to solve the market clearing problem.
