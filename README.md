# CCN Software Workshop at SfN, Nov 2025

Materials for CCN software workshop at the Society for Neuroscience Conference, November 2025.

We have a slack channel for communicating with attendees, if you haven't received an invitation, please send us a note!

> [!NOTE]
> The rest of this README is for contributors to the workshop.

## TODO

- [ ] decide where to put group projects on site
- [ ] hide stuff on the website? i.e., don't need to have full toctree for all three types of notebooks
- [ ] group projects should have "helpful links" section at the top, which link to relevant parts of the workshop, as well as documentation and other resources.
    - [ ] tell people about %whos magic
- [ ] add slide to plenoptic presentation talking about projects that use it (or could use it), including physiology (e.g., Neel's project)
- [ ] come up with pairs 
- [ ] make sure all crossref links are working
- [ ] update the links to the sklearn / place cell notebook
- [ ] update presentations, and add link to "Edoardo's presentation" in visual coding notebook

## Building the site locally

> [!WARNING]
> We still need `jupyterlab < 4.3` in order to get the link highlighting working. There's a class, `.jp-ThemedContainer`, which removes the link styling. I posted [a question about this issue](https://discourse.jupyter.org/t/link-highlighting-not-working/38053/3), and it turns out it's [an issue with jupyterlab_myst](https://github.com/jupyter-book/jupyterlab-myst/issues/248), which they're still working on.


To build the site locally, clone this repo and install it in a fresh python 3.12 environment (`pip install -e .`). Then run `make -C docs html O="-T"` and open `docs/build/html/index.html` in your browser.

When building the docs, you can use the `RUN_NB` environment variable to limit the notebooks that are executed. To do so, set it to a comma separated list of strings: any notebooks whose name is a substring of one of the strings will run. For example, `RUN_NB=current,head` will run `group_projects/01_head_direction.md` and `live_coding/02_current_injection.md`.

## Group projects

In this workshop, we are experimenting with group projects. We will have participants break up into groups of 4 or 5, each joined by an instructor or TA, and work through a notebook using `pynapple` and `nemos` to analyze a real neuroscience dataset.

Or maybe have them doing it in pairs? would avoid need for screens and mitigate quarterbacking risk

- Participants should work together, making sure that everyone understands what's going on and one person isn't doing all the coding or making all the decisions.
- Each participant should get practice writing the analysis code. How that happens is up to you: have everyone work in their own notebook, have one notebook that you take turns writing in, etc.
- Instructors / TAs are there to answer questions, either scientific or technical, but you are encouraged to try to solve things yourself, referring back to the materials presented in the workshops as well as the package documentation.
- At the end of the project time period, we will (**DECIDE**)
    - regroup and have one or two groups present their work to the class
    - find another group and swap notebooks, comparing their approach
    - form new groups of three(?) all from different groups, who talk through what they did
- if one of the second two, could have people also describe what their partner did for the whole class

Possibility:
- Have them work in pairs, grouped into pods of 2 or 3 pairs. Each pod has a single instructor / TA.
    - We should create pairs in advance, trying to match technical levels and scientific interests.
- Show slide of norms beforehand, which are basically the notes from above: everyone should spend time typing, everyone has something to contribute, this is cooperative, think out loud and consult each other.
- For sharing at end of session:
    - could do a "gallery" in pods: have one group talk through the steps in their notebook and ask if anyone has something different / interesting they did.
    - then come back as a big group and ask bigger meta questions

## strip_text.py

This script creates two copies of each file found at `docs/source/full/*/*md`, the copies are placed at `docs/source/users/*/*.md` and `docs/source/presenters/*/*.md`. Neither of these copies are run; the presenters version is intended as a reference for presenters, while the users version is what users will start with.

For this to work:
- The title should be on a line by itself, use `#` (e.g., `# My awesome title`) and be the first such line (so no comments above it).
- All headers must be markdown-style (using `#`), rather than using `------` underneath them.
- You may need to place blank newlines before/after any `div` opening or closing. I think if you don't place newlines after the `div` opening, it will consider everything after it part of a markdown block (which is probably not what you want if it's a `{code-cell}`).
- divs should not cover any headers: if a header happens in the middle of a div, you'll have to close right before hand and open a new one right after (this is because all headers are rendered in all versions).

Full notebook:
- Will not render any markdown wrapped in a div with `class='render-user'` or `class='render-presenter'` (but will render those wrapped in `class='render-all'`)
- Will not render or run any code wrapped in a div at all! Thus, for code that you want in all notebooks, add `:tag: [render-all]`, but for code that you only want in the user / presenter notebook, wrap it in a div with `class='render-user'` / `class='render-presenter'`. 
- Similarly, wrapping colon-fence blocks (which use `:::`, e.g., admonitions) are messed up when you wrap them in a `div`. But they have a `:class:` attribute themselves, so just add the appropriate `render` class there. See the "Download" admonition at the top of each notebook for an example.

Presenters version preserves:
- All markdown headers.
- All code blocks.
- Only colon-fence blocks (e.g., admonitions) that have the class `render-presenter` or `render-all`
- Only markdown wrapped in a `<div class='render-presenter'>` or `<div class='render-all'>`.

Users version preserves:
- All markdown headers.
- Only code blocks with `:tag: [render-all]` *OR* wrapped in a `<div class='render-user'>`. For code blocks in render-user divs, you should probably also add the `skip-execution` tag
- Only colon-fence blocks (e.g., admonitions) that have the class `render-user` or `render-all`
- Only markdown wrapped in a `<div class='render-user>` or `<div class='render-all'>`.

## Participants

We build all of the notebooks as a sphinx site, which we can browse. `index.md` is the main way in which participants will view the information present in this repo. They are instructed to clone this repo and run `scripts/setup.py`, which will download all the data, run `scripts/strip_text.py`, and convert the resulting user notebooks to `ipynb` files, placing them in the `notebooks/` directory. This is what users will see on their screen as they work on their notebook.

This means that you should check what these notebooks look like to make sure everything renders correctly. In particular, intersphinx **does not** work here (though it does work on the built website). Thus, you should instead use explicit links to the relevant documentation, where necessary. 
    
## binder

See [nemos Feb 2024 workshop](https://github.com/flatironinstitute/nemos-workshop-feb-2024) for details on how to set up the Binder
