# Design specification

## Input

The scientist provides a datafile, specifying which columns are physical state.
The scientist has a prior intuition to how many phases there and what is the underlying manifold dimension.

| p | T | rho_1 | h_1 | rho_2 | h_2 | ... | S | other data |
|---|---|---    |---  |---    | --- | --- |---| --- |
| x | x | x     | x   | x     | x   | ... | x | x   |

## Output

The ultimate output will be a module that implements the equation of state to be imported by the flow simulator.
The module contains the string tags of the state variables from the input data, partitioned into the state-state and the extra-curve-fits.

The models are compiled into C code. The methods needed are:
- encode a state (throw an error if input is invalid)
- decode a state 
- provide tangents to the state w.r.t. the latent unknowns
- throw an error if outside of training region
- return the other outputs from a latent unknown (S_gas, phase, etc)

The scientist fills in the documentation.

The training system logs the final hyperparameters and training information (timestamps etc.) in the final module.

## Process

The scientist gives initial guesses for phases, number of barriers, final order of the polynomials, and which architecture to use. 

The system then performs the autoencoder training.
Hyperparameter tuning happens under-the-hood.
The training system is not neccessarily running on the scientist's computer.

Baseline simulations are run during training to be shown to the scientist. The final model must be able to pass certain basic tests. 

The scientist is shown renderings of the final interpolated surface.
The identified phases are highlighted. The scientist is given the option to label the phases.
The phases are meaningless to the underlying simulation and are only there for human interpretation of output.


## Underlying implementation


## EOS Enhancement

One EOS will be a ground-truth boot-strapper to another EOS.

The scientist takes a prebuilt one (e.g. pure water) and provides auxilary data (e.g. water + a new gas).
The scientist specifies which fields are old and which are new.
The scientist specifies how many new dimensions are needed in the latent space.

The system loads the previous network, freezes the variables, and appends new curves and coefficients.
Training resumes.
The previous dataset is used as a regression test to assert the new EOS reproduces the previous one exactly.

The latent space is strictly appended to:
\begin{equation}
q_{new} = \{ q_{old}, 0 \}\quad\text{on}\quad \Omega_{old}
\end{equation}