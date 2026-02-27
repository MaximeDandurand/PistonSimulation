# PistonSimulation
## How to run
Run SimulationGraph to visualize a single config. You can change the config at the end of the file.  
Run VisualizeOptimizer to run and visualize a gradient search in a grid. It runs multiple instances to visualize the convergence.  
Run GridSearch to run a grid search(can take a long time).  

## Objective function

Both search uses the same scoring solution in EvaluationEngine.  
The GradientOptimizer also add penalty based on how much the result is invalid.