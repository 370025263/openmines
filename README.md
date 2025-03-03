# OpenMines: A Light and Comprehensive Mining Simulation Environment for Truck Dispatching  
Accepted in: 2024 35th IEEE Intelligent Vehicles Symposium (IV)  
Paper: [http://arxiv.org/abs/2404.00622](http://arxiv.org/abs/2404.00622)  

**Authors**: Shi Meng<sup>1</sup>, Bin Tian<sup>2,∗</sup>, Xiaotong Zhang<sup>3</sup>, Shuangying Qi<sup>4</sup>, Caiji Zhang<sup>5</sup>, Qiang Zhang<sup>6</sup>

## Table of Contents
1. [Description](#description)  
2. [Installation](#installation)  
3. [Usage](#usage)  
    1. [Create a Mine Configuration](#create-a-mine-configuration)  
    2. [Create a Dispatch Algorithm](#create-a-dispatch-algorithm)  
    3. [Run the Simulation](#run-the-simulation)  
    4. [Visualize the Result](#visualize-the-result)  
4. [How to Write a New Dispatch Algorithm](#how-to-write-a-new-dispatch-algorithm)  
5. [OpenMines Objects and Available Properties (Reference)](#openmines-objects-and-available-properties-reference)  
6. [More Command Line Usage](#more-command-line-usage)  

## Description
OpenMines is a Python-based simulation environment designed for truck dispatching in mining operations. It provides a flexible and extensible framework to model and simulate various mining scenarios from a complex-system perspective with probabilistic user-defined events, enabling researchers and practitioners to evaluate and compare different dispatching algorithms.

Visualization is supported:  
![demo](./imgs/openminesdemo.gif)

## Installation
OpenMines is available on PyPI and can be installed using pip:
```shell
pip install openmines
```

## Usage

### 1. Create a Mine Configuration
First, you need to configure the mine by creating a JSON file. OpenMines provides example configuration files in the openmines/src/conf/ folder.

Example: `north_pit_mine.json` (Configuration for the North Pit Mine in Holingol, Inner Mongolia, China, with anonymized data)
```python
{
  "mine": {
    "name": "NorthPitMine"
  },
  "dispatcher": {
    "type": ["NaiveDispatcher", "RandomDispatcher", "NearestDispatcher", "FixedGroupDispatcher", "SPTFDispatcher", "SQDispatcher"]
  },
  "charging_site": {
    "name": "NorthPitMineChargingSite",
    "position": [0, 0],
    "trucks": [
      {"type": "OfficalTruck", "count": 9, "capacity": 77, "speed": 25},
      {"type": "CLTruck", "count": 29, "capacity": 35, "speed": 25},
      {"type": "XHTruck", "count": 33, "capacity": 55, "speed": 25}
    ]
  },
  ...
}
```

### 2. Create a Dispatch Algorithm
You can configure the dispatch algorithm in the `algo` folder (`openmines/src/dispatch_algorithms/`).  
The class should inherit from `BaseDispatcher` and implement the following methods:  
`give_init_order`, `give_haul_order`, `give_back_order` and the property `name`.

For example, the following code is a naive dispatch algorithm that always gives the first load site to the truck.
```python
## openmines/src/dispatch_algorithms/naive_dispatch.py
from __future__ import annotations
from openmines.src.dispatcher import BaseDispatcher

class NaiveDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NaiveDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0
```

### 3. Run the Simulation
You can run the simulation with the following command:
```shell
openmines -f <config_file>
# or
openmines run -f <config_file>
```
After the simulation, you can find the simulation ticks in the `$CWD/result` folder.  
The result folder will contain the following files:
- **MINE:{mine_name}_ALGO:{algo_name}_TIME:{sim_time}.json**: the mine information [the ticks]  
- {mine_name}_table.tiff [a performance table of the algorithms configured in your config]  
- {mine_name}.tiff [a production curve of the algorithms over time]  
![curve](./imgs/north_pit_mine.png)  
![table](./imgs/north_pit_mine_table.png)

### 4. Visualize the Result
You can visualize the result with the following command:
```shell
openmines -v <result_tike_file>.json
# or 
openmines visualize -f <result_tike_file>.json
```
The result will be a gif file in the `$CWD/result` folder.  
![snapshot](./imgs/snapshot.png)

---

### 5. How to Write a New Dispatch Algorithm

To write a new dispatch algorithm, you need to inherit from the `BaseDispatcher` class and implement the following three core methods:
```python
## openmines/src/dispatch_algorithms/naive_dispatch.py
from __future__ import annotations
from openmines.src.dispatcher import BaseDispatcher

class NaiveDispatcher(BaseDispatcher):
    def __init__(self):
        super().__init__()
        self.name = "NaiveDispatcher"

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        return 0
```

---

## 6. OpenMines Objects and Available Properties (Reference)

To help new dispatch algorithm developers quickly get started, here is a list of core objects and their available properties that are commonly used when writing dispatch strategies. These objects are typically accessible through the `mine` object.

### 6.1 Mine
**Description**: `Mine` is the core object of the simulation, containing global information of the mine.  
**Common Properties**:
- `load_sites`: `List[LoadSite]`  
  *(List of all load site objects)*
- `dump_sites`: `List[DumpSite]`  
  *(List of all dump site objects)*
- `charging_site`: `ChargingSite`  
  *(Charging site object)*
- `road`: `Road`  
  *(Road network object)*
- `dispatcher`: `BaseDispatcher`  
  *(Dispatcher object)*
- `trucks`: `List[Truck]`  
  *(List of all truck objects)*
- `produced_tons`: `float`  
  *(Total production of the mine)*
- `service_count`: `int`  
  *(Total number of completed orders)*

### 6.2 LoadSite
**Description**: Load site object, containing multiple shovels.  
**Common Properties**:
- `shovel_list`: `List[Shovel]`  
  *(List of shovels in the load site)*
- `parkinglot`: `ParkingLot`  
  *(Parking lot object)*
- `service_ability_ratio`: `float`  
  *(Service ability ratio, affected by shovel downtime)*
- `estimated_queue_wait_time`: `float`  
  *(Estimated earliest service waiting time for the shovels)*
- `avg_queue_wait_time`: `float`  
  *(Average waiting time for the shovels)*
- `load_site_productivity`: `float`  
  *(Load site productivity, tons per minute)*
- `status`: `Dict[str, str]`  
  *(Historical status records of the load site, including production and service count)*

### 6.3 Shovel
**Description**: Shovel object, responsible for loading minerals.  
**Common Properties**:
- `load_site`: `LoadSite`  
  *(Associated load site)*
- `shovel_tons`: `float`  
  *(Shovel bucket capacity)*
- `shovel_cycle_time`: `float`  
  *(Shovel cycle time in minutes)*
- `service_count`: `int`  
  *(Number of services performed)*
- `last_service_time`: `float`  
  *(Last service start time)*
- `last_service_done_time`: `float`  
  *(Last service end time)*
- `est_waiting_time`: `float`  
  *(Estimated waiting time)*
- `last_breakdown_time`: `float`  
  *(Last breakdown time)*
- `status`: `Dict[str, str]`  
  *(Historical status records for the shovel)*

### 6.4 DumpSite
**Description**: Dump site object, used for unloading minerals.  
**Common Properties**:
- `dumper_list`: `List[Dumper]`  
  *(List of unloading devices)*
- `truck_visits`: `int`  
  *(Total number of truck visits to the dump site)*
- `produce_tons`: `float`  
  *(Total production of the dump site)*
- `service_count`: `int`  
  *(Number of services performed)*

### 6.5 Dumper
**Description**: Dumper object, used for unloading.  
**Common Properties**:
- `dump_site`: `DumpSite`  
  *(Associated dump site)*
- `dump_time`: `float`  
  *(Dumper unloading time)*
- `dumper_tons`: `float`  
  *(Total tonnage unloaded by the dumper)*
- `service_count`: `int`  
  *(Number of services performed)*

### 6.6 Road
**Description**: Road network object, containing distance information between locations.  
**Common Properties**:
- `l2d_road_matrix`: `np.ndarray`  
  *(Load site to dump site distance matrix)*
- `d2l_road_matrix`: `np.ndarray`  
  *(Dump site to load site distance matrix)*
- `charging_to_load_road_matrix`: `List[float]`  
  *(Charging site to load site distance list)*
- `road_repairs`: `Dict[Tuple[Union[LoadSite, DumpSite], Union[LoadSite, DumpSite]], Tuple[bool, float]]`  
  *(Road repair status dictionary, where the key is a tuple of start and end points, and the value is a tuple of (repair status, expected repair completion time))*

**Common Methods**:
- `truck_on_road(start: Union[LoadSite, DumpSite], end: Union[LoadSite, DumpSite]) -> List[Truck]`  
  *(Get a list of trucks on the specified road)*

---

## 7. More Command Line Usage

In addition to the "run/visualize" commands mentioned earlier, `openmines` also supports the following commands to meet more advanced functional needs in research or experimental environments.

### 7.1 Fleet Scale Ablation Experiment for Single Scene
Test the production effect of the algorithm by varying truck scales in the same scene (configuration file):
```shell
openmines scene_based_fleet_ablation -f <config_file> -m <min_truck> -M <max_truck>
```
Example:
```shell
openmines scene_based_fleet_ablation -f my_mine_config.json -m 10 -M 50
```
This command will sample multiple truck scale points in the `[min_truck, max_truck]` range, run all the dispatch algorithms declared in your configuration, and output production comparisons or generate relevant charts.

### 7.2 Multi-Scene & Dual Algorithm Ablation Experiment
Compare the performance ratio of a target algorithm with the baseline algorithm across multiple scenes:
```shell
openmines algo_based_fleet_ablation -d <config_dir> -b <baseline_algo> -t <target_algo> -m <min_truck> -M <max_truck>
```
Example:
```shell
openmines algo_based_fleet_ablation -d configs/ -b NaiveDispatcher -t MySmartDispatcher -m 10 -M 100
```
This command will iterate through all the `.json` configuration files in the `configs/` folder, run the `baseline_algo` and `target_algo` with various fleet sizes (from `-m` to `-M`), and output comparison results or generate charts.

### 7.3 Log Analysis Commands
Analyze log files or directories, extract and generate summary reports:
```shell
openmines -a <log_path> 
# or
openmines analyze <log_path> [-d <dispatcher_name>]
```
Example:
```shell
# Analyze a specified log file
openmines -a /path/to/simulation.log

# Analyze the latest log file in the specified directory and specify the dispatcher name to query
openmines analyze /path/to/log/folder -d NaiveDispatcher
```
These commands will automatically identify the log content, extract and analyze key dispatch-related data, and output the generated report to the current working directory.

## 8. Reinforcement Learning Support

Reninforcement learning is a promising approach to achieve better performace on both flexibility and more.

Openmines managed to itegrate the truck-dispatching problem with the gymnasium standard.

```python

import gymnasium as gym

# Create environment
env = gym.make('mine/Mine-v1-dense', config_file="./conf/north_pit_mine.json")  # or mine/Mine-v1-sparse

# Reset environment
obs, info = env.reset()

# Run an episode
for _ in range(1000):
    # Execute using suggested action
    obs, reward, done, truncated, info = \
       env.step(info["sug_action"])
    
    if done or truncated:
        break

env.close()

```