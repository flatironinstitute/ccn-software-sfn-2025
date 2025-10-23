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

This notebook can be downloaded as **{nb-download}`place_cells_part_2.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model selection - Part 2

<div class="render-all">

In our previous analysis of the place field hyppocampal dataset we compared multiple encoding models and tried to figure out which predictor (position, speed or phase) had more explanatory power. In this notebook we will keep going on that effort and learn more principled (and convenient) approaches to model comparison combining NeMoS and scikit-learn.

## Learning Objectives

- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection
- 
</div>

## Scikit-learn

### How to know when to regularize?

In the [head direction](./head_direction.md) notebook, we fit the all-to-all connectivity of the head-tuning dataset using the Ridge regularizer, and we learned that regularization can combat overfitting. What we didn't show is how to choose a proper regularizer. Generally, too much regularization leads to underfitting, i.e. the model is too simple and doesn't capture the neural variability well. To little regularization may overfit, especially when we have a large number of parameters, i.e. out model will capture both signal and noise. This is what we saw in the head direction notebook when we used the raw spike history as predictor. 

What we are looking for is a regularization strength that balances out the bias towards simpler models with the variance necessary to explain the data. However, how do we know how much we should regularize? One thing we can do is use cross-validation to see whether model performance on unseen data improves with regularization (behind the scenes, this is what we did!). We'll walk through how to do that now.

Instead of implementing our own cross-validation machinery, the developers of nemos decided that we should write the package to be compliant with [scikit-learn](https://scikit-learn.org), the canonical machine learning python library. Our models are all what scikit-learn calls "estimators", which means they have `.fit`, `.score.` and `.predict` methods. Thus, we can use them with scikit-learn's objects out of the box.

We're going to use scikit-learn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object, which performs a cross-validated grid search, as [Edoardo explained in his presentation](https://users.flatironinstitute.org/~wbroderick/presentations/sfn-2025/model_selection.pdf).

This object requires an estimator, our `glm` object here, and `param_grid`, a dictionary defining what to check. For now, let's just compare Ridge regularization with no regularization:

<div class="render-user render-presenter">

- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.

</div>

```{code-cell} ipython3
# define a Ridge GLM
glm = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)
param_grid = {
    "regularizer_strength": [0.0001, 1.],
}
```

<div class="render-user render-presenter">

- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(glm, param_grid, cv=cv_folds)
cv
```

This will take a bit to run, because we're fitting the model many times!

<div class="render-user render-presenter">

- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:
</div>

```{code-cell} ipython3
cv.fit(X, count)
```

<div class="render-user render-presenter">

- We got a warning because we didn't specify the regularizer strength, so we just fell back on default value.
- Let's investigate results:
</div>

Cross-validation results are stored in a dictionary attribute called `cv_results_`, which contains a lot of info. Let's convert that to a pandas dataframe for readability,

```{code-cell} ipython3
import pandas as pd

pd.DataFrame(cv.cv_results_)
```

The most informative for us is the `'mean_test_score'` key, which shows the average of `glm.score` on each test-fold. Thus, higher is better, and we can see that the UnRegularized model performs better.


### Select basis

We can do something similar to select the basis. In the above example, I just told you which basis function to use and how many of each. But, in general, you want to select those in a reasonable manner. Cross-validation to the rescue!

Unlike the glm objects, our basis objects are not scikit-learn compatible right out of the box. However, they can be made compatible by using the `.to_transformer()` method (or, equivalently, by using the `TransformerBasis` class)

<div class="render-user render-presenter">

- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:

</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position").to_transformer()
# or equivalently:
position_basis = nmo.basis.TransformerBasis(nmo.basis.MSplineEval(n_basis_funcs=10, label="position"))
position_basis
```

This gives the basis object the `transform` method, which is equivalent to `compute_features`. However, transformers have some limits:

<div class="render-user render-presenter">

- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:

</div>

```{code-cell} ipython3
:tags: [raises-exception]

position_basis.transform(position)
```

<div class="render-user render-presenter">

- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.

</div>

Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.

```{code-cell} ipython3
position_basis.transform(position[:, np.newaxis])
```

<div class="render-user render-presenter">

- If the basis is composite (for example, the addition of two 1D bases), the transformer will expect a shape of `(n_sampels, 1)` each 1D component. If that's not the case, you need to call `set_input_shape`:

</div>

If the basis has more than one component (for example, if it is the addition of two 1D bases), the transformer will expect an input shape of `(n_sampels, 1)` pre component. If that's not the case, you'll provide a different input shape by calling `set_input_shape`.

**Option 1)** One input per component:

```{code-cell} ipython3
# generate a composite basis
basis_2d = nmo.basis.MSplineEval(5) +  nmo.basis.MSplineEval(5)
basis_2d = basis_2d.to_transformer()

# this will work: 1 input per component
x, y = np.random.randn(10, 1), np.random.randn(10, 1)
X = np.concatenate([x, y], axis=1)
result = basis_2d.transform(X)
```

**Option 2)** Multiple input per component.

<div class="render-user render-presenter">

- Then you can call transform on the 2d input as expected.
</div>

```{code-cell} ipython3
# Assume 2 input for the first component and 3 for the second.
x, y = np.random.randn(10, 2), np.random.randn(10, 3)
X = np.concatenate([x, y], axis=1)
try:
    basis_2d.transform(X)
except Exception as e:
    print("Exception Raised:")
    print(repr(e))

# Set the expected input shape instead.

# array
res1 = basis_2d.set_input_shape(x, y).transform(X)
# int
res2 = basis_2d.set_input_shape(2, 3).transform(X)
# tuple
res3 = basis_2d.set_input_shape((2,), (3,)).transform(X)
```

<div class="render-all">

- You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:

</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed").to_transformer().set_input_shape(1)
basis = position_basis + speed_basis
basis
```

Let's create a single TsdFrame to hold all our inputs:

<div class="render-user render-presenter">

- Create a single TsdFrame to hold all our inputs:
</div>

```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position, speed], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```

<div class="render-user render-presenter">

- Pass this input to our transformed additive basis:
</div>

Our new additive transformer basis can then take these behavioral inputs and turn them into the model's design matrix.

```{code-cell} ipython3
basis.transform(transformer_input)
```

### Pipelines

We need one more step: scikit-learn cross-validation operates on an estimator, like our GLMs. if we want to cross-validate over the basis or its features, we need to combine our transformer basis with the estimator into a single estimator object. Luckily, scikit-learn provides tools for this: pipelines.

Pipelines are objects that accept a series of (0 or more) transformers, culminating in a final estimator. This is defined as a list of tuples, with each tuple containing a human-readable label and the object itself:

<div class="render-user render-presenter">

- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!

</div>

```{code-cell} ipython3
# set the reg strength to the optimal
glm = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs={"tol": 10**-12})
pipe = pipeline.Pipeline([
    ("basis", basis),
    ("glm", glm)
])
pipe
```

This pipeline object allows us to e.g., call fit using the *initial input*:

<div class="render-user render-presenter">

- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:
</div>

```{code-cell} ipython3
pipe.fit(transformer_input, count)
```

We then visualize the predictions the same as before, using `pipe` instead of `glm`.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
visualize_model_predictions(pipe, transformer_input)
```

### Cross-validating on the basis

<div class="render-all">

Now that we have our pipeline estimator, we can cross-validate on any of its parameters!

</div>

```{code-cell} ipython3
pipe.steps
```

Let's cross-validate on the number of basis functions for the position basis, and the identity of the basis for the speed. That is:

<div class="render-user render-presenter">

Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed
</div>

```{code-cell} ipython3
print(pipe["basis"]["position"].n_basis_funcs)
print(pipe["basis"]["speed"])
```

For scikit-learn parameter grids, we use `__` to stand in for `.`:

<div class="render-user render-presenter">

- Construct `param_grid`, using `__` to stand in for `.`
- In sklearn pipelines, we access nested parameters using double underscores:
  - `pipe["basis"]["position"].n_basis_funcs` ← normal Python syntax
  - `"basis__position__n_basis_funcs"` ← sklearn parameter grid syntax

</div>

```{code-cell} ipython3
param_grid = {
    "basis__position__n_basis_funcs": [5, 10, 20],
    "basis__speed": [nmo.basis.MSplineEval(15).set_input_shape(1),
                      nmo.basis.BSplineEval(15).set_input_shape(1),
                      nmo.basis.RaisedCosineLinearEval(15).set_input_shape(1)],
}
```

<div class="render-user render-presenter">

- Cross-validate as before:
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

<div class="render-user render-presenter">

- Investigate results:
</div>

```{code-cell} ipython3
pd.DataFrame(cv.cv_results_)
```

scikit-learn does not cache every model that it runs (that could get prohibitively large!), but it does store the best estimator, as the appropriately-named `best_estimator_`.

<div class="render-user render-presenter">

- Can easily grab the best estimator, the pipeline that did the best:
</div>

```{code-cell} ipython3
best_estim = cv.best_estimator_
best_estim
```

We then visualize the predictions of `best_estim` the same as before.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(best_estim, transformer_input)
```

### Feature selection

Now, finally, we understand almost enough about how scikit-learn works to figure out whether both position and speed are necessary inputs, i.e., to do feature selection. 

What we would like to do here is comparing alternative models: position + speed, position only or speed only. However, scikit-learn's cross-validation assumes that the input to the pipeline does not change, only the hyperparameters do. So, how do we go about model selection since we require different input for different model we want to compare?

Here is a neat NeMoS trick to circumvent that. scikit-learn's GridSearchCV assumes the INPUT stays the same across all models, but for feature selection, we want to compare models with different features (position + speed, position only, speed only). The solution: create a "null" basis that produces zero features, so all models take the same 2D input (position, speed) but some features become empty. First we need to define this "null" basis taking advantage of `CustomBasis`, which defines a basis from a list of functions.

```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))

# Create a null basis using the custom basis class
null_basis = nmo.basis.CustomBasis([func]).to_transformer()

# this creates an empty feature
null_basis.compute_features(position).shape
```

Why is this useful? Because we can use this `null_basis` and basis composition to do model selection.

```{code-cell} ipython3
# first we note that the position + speed basis is in the basis attribute
print(pipe["basis"].basis)

position_bas = nmo.basis.MSplineEval(n_basis_funcs=10).to_transformer()
speed_bas = nmo.basis.MSplineEval(n_basis_funcs=15).to_transformer()

# define 2D basis per each model 
basis_all = position_bas + speed_bas
basis_position = position_bas + null_basis
basis_speed = null_basis + speed_bas

# assign label (not necessary but nice)
basis_all.label = "position + speed"
basis_position.label = "position"
basis_speed.label = "speed"


# then we create a parameter grid defining a grid of 2D basis for each model of interest
param_grid = {
    "basis__basis": 
    [
        basis_all,  
        basis_position, 
        basis_speed 
    ],
}

# finally we define and fit our CV
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

Let's now take a look to the model results.

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)

# let's just plot a minimal subset of cols
cv_df[["param_basis__basis", "mean_test_score", "rank_test_score"]]
```
Unsurprisingly, position comes up as the predictor with the larger explnatory power and speed adds marginal benefits.

For the next project, you can use all the tools showcased here to find a better encoding model model for these hyppocampal neurons. 


## Conclusion

Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)

## References

<div class="render-all">

The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

</div>

```{code-cell} ipython3

```
