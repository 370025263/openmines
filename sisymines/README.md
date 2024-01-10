# sisymines
an open-pit mine traffic simulator for mining truck dispatch algorithms

## Installation
```shell
pip install .
```
 
## Usage
### 1. Create a mine
you need first configurate the mine in the config file.
sisymines offer example config files in the config folder ( (sisymines/src/conf/)

### 2. Create a dispatch algorithm
you can configurate the dispatch algorithm in the algo folder (sisymines/src/dispatch_algorithms/)
the class should inherit from BaseDispatcher and implement the following methods:
**give_init_order, give_haul_order, give_back_order** and property **name**.

For example, the following code is a naive dispatch algorithm that always give the first load site to the truck.
```python
## sisymines/src/dispatch_algorithms/naive_dispatch.py
from __future__ import annotations
from sisymines.src.dispatcher import BaseDispatcher



class NaiveDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NaiveDispatcher"  # IMPORTANT: set the name of the algorithm; this will be used in the config file

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个load site开始
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个load site开始
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        # 从第一个dump site开始
        return 0


```
## 3. Run the simulation

then you can run the simulation with the following command:
```shell
sisymines -f <config_file>
# or
sisymines run -f <config_file>
```
after the simulation, you can find the simulation ticks in the $CWD/result folder.
the result folder will contain the following files:
- **mine.csv**: the mine information  [todo]
- MINE:YOUR_MINE_NAME ALGO:NaiveDispatcher_TIME:2024-01-10 17:57:08.json [the ticks]

## 4. Visualize the result
you can visualize the result with the following command:
```shell
sisymines -v <result_file>
# or 
sisymines visualize -f <result_file>
```
the result will be a gif file in the $CWD/result folder.