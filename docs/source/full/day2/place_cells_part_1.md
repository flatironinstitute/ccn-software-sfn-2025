---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input, render-all]

%load_ext autoreload
%autoreload 2

%matplotlib inline
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`place_cells.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model selection

In this notebook we will keep working on the hippocampal place field recordings. In particular, we will learn how to model neural responses to multiple predictors: position, speed and theta phase. We will also learn how we can leverage GLM to assess which predictor is more informative for explaining the observed activity.

<div class="render-user">
Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track, which we [explored yesterday](../day1/phase_precession-users.md).
</div>

<div class="render-all">

## Learning objectives

- Learn how to combine NeMoS basis objects for modeling multiple predictors
- Learn how to score and compare models.

</div>

```{code-cell} ipython3
:tags: [render-all]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure plots some
plt.style.use(nmo.styles.plot_style)

import workshop_utils

from sklearn import model_selection
from sklearn import pipeline

# shut down jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

```

## Pynapple

<div class="render-user render-presenter">
- Load the data using pynapple.
</div>

```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
data
```

<div class="render-user render-presenter">
- Extract the spike times and mouse position.
</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = data["units"]
position = data["position"]
```

For today, we're only going to focus on the times when the animal was traversing the linear track. 
This is a pynapple `IntervalSet`, so we can use it to restrict our other variables:

<div class="render-user render-presenter">

- Restrict data to when animal was traversing the linear track.

</div>

```{code-cell} ipython3
:tags: [render-all]

position = position.restrict(data["forward_ep"])
spikes = spikes.restrict(data["forward_ep"])
```

The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells with firing above 0.3 Hz.

<div class="render-user render-presenter">

- Restrict neurons to only excitatory neurons, discarding neurons with a low-firing rate.

</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = spikes.getby_category("cell_type")["pE"]
spikes = spikes.getby_threshold("rate", 0.3)
```

### Place fields

By plotting the neuronal firing rate as a function of position, we can see that these neurons are all tuned for position: they fire in a specific location on the track.

<div class="render-user render-presenter">

- Visualize the *place fields*: neuronal firing rate as a function of position.
</div>

```{code-cell} ipython3
:tags: [render-all]

place_fields = nap.compute_tuning_curves(spikes, position, bins=50, epochs=position.time_support, feature_names=["distance"])
workshop_utils.plot_place_fields(place_fields)
```

To decrease computation time, we're going to spend the rest of the notebook focusing on the neurons highlighted above. We're also going to bin spikes at 100 Hz and up-sample the position to match that temporal resolution.

<div class="render-user render-presenter">

- For speed, we're only going to investigate the three neurons highlighted above.
- Bin spikes to counts at 100 Hz.
- Interpolate position to match spike resolution.

</div>

```{code-cell} ipython3
:tags: [render-all]

neurons = [82, 92, 220]
place_fields = place_fields.sel(unit=neurons)
spikes = spikes[neurons]
bin_size = .01
count = spikes.count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
print(count.shape)
print(position.shape)
```

### Speed modulation


The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze. In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

<div class="render-user render-presenter">

- Compute animal's speed for each epoch.

</div>

```{code-cell} ipython3
:tags: [render-all]

speed = []
# Analyzing each epoch separately avoids edge effects.
for s, e in position.time_support.values: 
    pos_ep = position.get(s, e)
    # Absolute difference of two consecutive points
    speed_ep = np.abs(np.diff(pos_ep)) 
    # Padding the edge so that the size is the same as the position/spike counts
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") 
    # Converting to cm/s 
    speed_ep = speed_ep * position.rate
    speed.append(speed_ep)

speed = nap.Tsd(t=position.t, d=np.hstack(speed), time_support=position.time_support)
print(speed.shape)
```

Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves):

<div class="render-user render-presenter">

- Compute the tuning curve with pynapple's [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves)

</div>

```{code-cell} ipython3
:tags: [render-all]

tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])
```

<div class="render-user render-presenter">

- Visualize the position and speed tuning for these neurons.
</div>

```{code-cell} ipython3
:tags: [render-all]

fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```

These neurons show a strong modulation of firing rate as a function of speed but we also notice that the animal, on average, accelerates when traversing the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speeds? We can use NeMoS to model the activity and give the position and the speed as input variable.

<div class="render-user render-presenter">

These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. We're going to build a GLM to predict neuronal firing rate -- which variable should we use? Is the speed tuning just epiphenomenal?

</div>

## NeMoS

(basis_eval_place_cells)=
### Basis evaluation

As we've seen before, we will use basis objects to represent the input values.  In previous tutorials, we've used the `Conv` basis objects to represent the time-dependent effects we were looking to capture. Here, we're trying to capture the non-linear relationship between our input variables and firing rate, so we want the `Eval` objects. In these circumstances, you should look at the tuning you're trying to capture and compare to the [basis kernels (visualized in NeMoS docs)](table_basis): you want your tuning to be capturable by a linear combination of them.

In this case, several of these would probably work; we will use [`MSplineEval`](nemos.basis.MSplineEval) for both, though with different numbers of basis functions.

Additionally, since we have two different inputs, we'll need two separate basis objects.

:::{note}

Later in this notebook, we'll show how to cross-validate across basis identity, which you can use to choose the basis.

:::

<div class="render-presenter">

- why basis?
   - without basis:
     - either the GLM says that firing rate increases exponentially as position or speed increases, which is fairly nonsensical,
     - or we have to fit the weight separately for each position or speed, which is really high-dim
   - so, basis allows us to reduce dimensionality, capture non-linear modulation of firing rate (in this case, tuning)
- why eval?
    - basis objects have two modes:
    - conv, like we've seen, for capturing time-dependent effects
    - eval, for capturing non-linear modulation / tuning
- why MSpline?
    - when deciding on eval basis, look at the tuning you want to capture, compare to the kernels: you want your tuning to be capturable by a linear combination of these
    - in cases like this, many possible basis objects we could use here and what I'll show you in a bit will allow you to determine which to use in principled manner
    - MSpline, BSpline, RaisedCosineLinear : all would let you capture this
    - weird choices:
        - cyclic bspline, except maybe for position? if end and start are the same
        - RaisedCosineLog (don't want the stretching)
        - orthogonalized exponential (specialized for...)
        - identity / history (too basic)
</div>


<div class="render-user render-presenter">

- Create a separate basis object for each model input.
- Provide a label for each basis ("position" and "speed")
- Visualize the basis objects.
</div>

```{code-cell} ipython3
:tag: [render-presenter]

position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

However, now we have an issue: in all our previous examples, we had a single basis object, which took a single input to produce a single array which we then passed to the `GLM` object as the design matrix. What do we do when we have multiple basis objects?

We could call `basis.compute_features()` for each basis separately and then concatenated the outputs, but then we have to remember the order we concatenated them in and that behavior gets unwieldy as we add more bases.

Instead, NeMoS allows us to combine multiple basis objects into a single "additive basis", which we can pass all of our inputs to in order to produce a single design matrix:

<div class="render-user render-presenter">

- Combine the two basis objects into a single "additive basis"
</div>

```{code-cell} ipython3
:tag: [render-presenter]

# equivalent to calling nmo.basis.AdditiveBasis(position_basis, speed_basis)
basis = position_basis + speed_basis
```

<div class="render-user render-presenter">

- Create the design matrix!
- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.
- `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).

</div>

```{code-cell} ipython3
:tag: [render-presenter]

X = basis.compute_features(position, speed)
X
```

### Model learning

As we've done before, we can now use the Poisson GLM from NeMoS to learn the combined model:

<div class="render-user render-presenter">

- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.

</div>

```{code-cell} ipython3
:tag: [render-presenter]


glm = nmo.glm.PopulationGLM(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

glm.fit(X, count)
```

### Prediction

Let's check first if our model can accurately predict the tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

<div class="render-user render-presenter">

- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!

</div>

```{code-cell} ipython3
:tag: [render-presenter]

# predict the model's firing rate
predicted_rate = glm.predict(X) / bin_size

# same shape as the counts we were trying to predict
print(predicted_rate.shape, count.shape)

# compute the position and speed tuning curves using the predicted firing rate.
glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=speed.time_support, feature_names=["speed"])
```

<div class="render-user render-presenter">

- Compare model and data tuning curves together. The model did a pretty good job!

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```

<div class="render-all">

We can see that this model does a good job capturing both the position and the speed. In the rest of this notebook, we're going to investigate all the scientific decisions that we swept under the rug: should we regularize the model? what basis should we use? do we need both inputs?

To make our lives easier, let's create a helper function that wraps the above
lines, because we're going to be visualizing our model predictions a lot.

</div>

```{code-cell} ipython3
:tag: [render-all]

def visualize_model_predictions(glm, X):
    # predict the model's firing rate
    predicted_rate = glm.predict(X) / bin_size

    # compute the position and speed tuning curves using the predicted firing rate.
    glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
    glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=position.time_support, feature_names=["speed"])

    workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```

# Model Comparison

Now that we have learned how to fit a model including both speed and position predictors, we would like compare the following models: 

- A model including all predictors
- A model including only position
- A model including only speed

To understand which of feature has more explanatory power (i.e. predicts best the neural activity).

For convenience, for each model configuration lets store the corresponding basis and its input(s) in two dictionaries.

<div class="render-user render-presenter">

- Define a dictionary of bases for computing the features for each of the following models:
  - "position + speed"
  - "position"
  - "speed"
- Define a dictionary with the input time series that will be processed the bases as a tuple, i.e. `features = {"position": (position,), "speed": (speed,)...}`

</div>


```{code-cell} ipython3
:tag: [render-presenter]

bases = {
    "position": position_basis,
    "speed": speed_basis,
    "position + speed": position_basis + speed_basis,
}

features = {
    "position": (position,),
    "speed": (speed,),
    "position + speed": (position, speed),
}

```

To avoid overfitting, we would like to train the model on a subset of the data, and score it on an independent subset. We will explore more sophisticated techniques later on, but for the purpose of this tutorial let's use 50% of the trials for training and 50% for scoring the model.

<div class="render-user render-presenter">

- Split the trial into a training and a test set, 50% each.

</div>

```{code-cell} ipython3
:tag: [render-presenter]

train_iset = position.time_support[::2] # Taking every other epoch
test_iset = position.time_support[1::2]
```

Now, let's construct our models, fit them and compute scores.

<div class="render-user render-presenter">

- Loop over model configuration. 
- Fit a GLM on the training set. 
- Score on the test set. You can use `score_type="pseduo-r2-McFadden"` for a log-likelihood based score that is normalized between 0 and 1 (the higher the better).
- Save the model scores in a dictionary.

</div>


```{code-cell} ipython3
:tag: [render-presenter]

scores = {}
predicted_rates = {}
models = {}

glm_kwargs = dict(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

for m in bases:
    print("1. Evaluating basis : ", m)
    X = bases[m].compute_features(*features[m])

    print("2. Fitting model : ", m)
    glm = nmo.glm.PopulationGLM(**glm_kwargs)
    glm.fit(
        X.restrict(train_iset),
        count.restrict(train_iset),
    )

    print("3. Scoring model : ", m)
    scores[m] = glm.score(
        X.restrict(test_iset),
        count.restrict(test_iset),
        score_type="pseudo-r2-McFadden",
    )

    # Store the model
    models[m] = glm

scores = pd.Series(scores)
scores = scores.sort_values()
scores
```

As we can see the model that ranks best includes both variables, while position seems to have more explanatory power. 

<div class="render-user render-presenter">

- Plot the scores as an error bar.

</div>

```{code-cell} ipython3
:tag: [render-presenter]

plt.figure(figsize=(5, 3))
plt.barh(np.arange(len(scores)), scores)
plt.yticks(np.arange(len(scores)), scores.index)
plt.xlabel("Pseudo r2")
plt.tight_layout()
```

As an exercise, try to include the theta phase as a predictor, and try to model the interaction between theta phase and position in the place field.

<div class="render-user render-presenter">

- Finally, let's visualize each model.

</div>

```{code-cell} ipython3
:tag: [render-all]

m = "position"
print(f"\n{'='*50}")
print(f"  {m}")
print(f"{'='*50}\n")

X = bases[m].compute_features(*features[m])
visualize_model_predictions(models[m], X)
    
```

```{code-cell} ipython3
:tag: [render-all]

m = "speed"
print(f"\n{'='*50}")
print(f"  {m}")
print(f"{'='*50}\n")

X = bases[m].compute_features(*features[m])
visualize_model_predictions(models[m], X)
    
```


```{code-cell} ipython3
:tag: [render-all]

m = "position + speed"
print(f"\n{'='*50}")
print(f"  {m}")
print(f"{'='*50}\n")

X = bases[m].compute_features(*features[m])
visualize_model_predictions(models[m], X)
    
```
Note that you can use basis multiplication to model interactions.