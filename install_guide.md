1. Clone monodepth repo
2. Add an empty `__init__.py` file to monodepth folder
3. On line 14 of `networks/depth_decoder.py` change `from layers import *` to `from ..layers import *`

4. Create python 3.6.6 venv (using conda or other)
5. Run `conda install -c conda-forge gcc=12.1.0 libgcc`
5. Install required packages `pip install -r requirements.txt`
6. Ubuntu users: install the following packages `sudo apt install libgtk2.0-dev pkg-config libstdc++6 -y`