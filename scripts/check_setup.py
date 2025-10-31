#!/usr/bin/env python3

try:
    from rich import print
except ImportError:
    print("rich not found. Try running pip install rich.")
    print("The following will not be pretty...")
import pathlib
import sys
import subprocess
import warnings
import os

repo_dir = pathlib.Path(__file__).parent.parent
os.environ["NEMOS_DATA_DIR"] = os.environ.get("NEMOS_DATA_DIR", str(repo_dir / "data"))

errors = 0

python_version = sys.version.split('|')[0]
if '3.12' in python_version:
    print(f":white_check_mark: Python version: {python_version}")
else:
    print(f":x: Python version: {python_version}. Create a new virtual environment.")
    errors += 1

try:
    import nemos
except ModuleNotFoundError:
    errors += 1
    print(":x: Nemos not found. Try running [bold]pip install nemos[/bold]")
else:
    print(f":white_check_mark: Nemos version: {nemos.__version__}")

try:
    import pynapple as nap
except ModuleNotFoundError:
    errors += 1
    print(":x: pynapple not found. Try running [bold]pip install pynapple[/bold]")
else:
    print(f":white_check_mark: pynapple version: {nap.__version__}")

p = subprocess.run(['jupyter', '--version'], capture_output=True)
if p.returncode != 0:
    errors += 1
    print(":x: jupyter not found. Try running [bold]pip install jupyter[/bold]")
else:
    # convert to str from bytestring
    stdout = '\n'.join(p.stdout.decode().split('\n')[1:])
    print(f":white_check_mark: jupyter found with following core packages:\n{stdout}")

p = subprocess.Popen(['jupyter', 'labextension', 'list'], stderr=subprocess.PIPE)
if os.name == "nt":
    search_cmd = "findstr"
else:
    search_cmd = "grep"
try:
    output = subprocess.check_output([search_cmd, 'myst'], stdin=p.stderr).decode().lower()
    p.wait()
except subprocess.CalledProcessError:
    errors += 1
    print(":x: jupyterlab_myst not found. Try running [bold]pip install jupyterlab_myst[/bold]")
else:
    if 'enabled' in output and 'ok' in output:
        with warnings.catch_warnings():
            # this import may give a deprecation warning about how jupyter handles paths
            warnings.simplefilter("ignore")
            import jupyterlab_myst
        print(f":white_check_mark: jupyterlab_myst version:\n{jupyterlab_myst.__version__}")
    else:
        errors += 1
        print(":x: jupyterlab_myst not set up correctly! Look at the output of `jupyter labextension list` and try running [bold]pip install jupyterlab_myst[/bold]")

repo_dir = pathlib.Path(__file__).parent.parent / 'notebooks'
gallery_dir = pathlib.Path(__file__).parent.parent / 'docs' / 'source' / 'full'
nbs = list(repo_dir.glob('**/*ipynb'))
gallery_scripts = [nb for nb in list(gallery_dir.glob('**/*md'))
                   if 'checkpoint' not in nb.name]
missing_nb = [f.stem for f in gallery_scripts
              if not any([f.stem == nb.stem.replace('-users', '') for nb in nbs])]
# index isn't a notebook, so don't check for it
missing_nb = [f for f in missing_nb if f != "index"]
if len(missing_nb) == 0:
    print(":white_check_mark: All notebooks found")
else:
    errors += 1
    print(f":x: Following notebooks missing: {', '.join(missing_nb)}")
    print("   Did you run [bold]python scripts/setup.py[/bold]?")

try:
    import workshop_utils
except ModuleNotFoundError:
    errors += 1
    print(f":x: workshop utilities not found. Try running [bold]pip install .[/bold] from the github repo.")
else:
    missing_files = []
    with warnings.catch_warnings():
        # this import may give warning about documentation_utils
        warnings.simplefilter("ignore")
        from nemos.fetch.fetch_data import _create_retriever
    retriever = _create_retriever()
    for f in workshop_utils.DOWNLOADABLE_FILES:
        # as far as I could find, retriever doesn't have a "check if file is downloaded"
        # function. (is_available just checks the *url* is available)
        if not (retriever.abspath / f).exists():
            missing_files.append(f)
    if len(missing_files) > 0:
        errors += 1
        print(f":x: Following data files not downloaded: {', '.join(missing_files)}")
        print("   Did you run [bold]python scripts/setup.py[/bold]?")
    else:
        print(":white_check_mark: All data files found!")


if errors == 0:
    print("\n:tada::tada: Congratulations, setup successful!")
    print("\nPlease run `jupyter lab notebooks/day2/current_injection-users.ipynb`, ")
    print("and ensure that you can run the first few cells (up until the cell containing ")
    print("`path = workshop_utils.fetch_data(\"allen_478498617.nwb\")`).")
else:
    print(f"\n:worried: [red bold]{errors} Errors found.[/red bold]\n")
    print("Unfortunately, your setup was unsuccessful.")
    print("Try to resolve following the suggestions above.")
    print("If you encountered many installation errors, run [bold] pip install .[/bold] (note the dot!)")
    print("If you are unable to fix your setup yourself, please come to the setup help")
    print("at the Omni San Diego Hotel, Gallery 1, from noon to 6pm on Wednesdsay, November 12 ")
    print("and show us the output of this command.")
