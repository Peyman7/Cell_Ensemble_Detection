# README: Cell Assembly Detection with Parallel Processing on Compute Canada

## Overview
This project implements **cell assembly detection** using neural spike train data on the Compute Canada HPC cluster. The workflow combines the computational power of **Apache Spark** for distributed processing and **Slurm** for resource management, enabling scalable and efficient analysis of large-scale neural datasets.

The pipeline processes neural spike train data using the **cell assembly detection** method to identify neuronal ensembles at multiple timescales and with arbitrary lag constellations. Tasks are distributed across nodes and CPUs, leveraging Apache Spark and Slurm to reduce runtime and improve fault tolerance.

- **Cell Assembly Detection (CAD)**: Extracts cell ensembles using CAD (Russo, Eleonora, and Daniel Durstewitz. "Cell assemblies at multiple time scales with arbitrary lag constellations." Elife 6 (2017): e19428).
- **High-Performance Computing Pipeline**: Developed a robust HPC pipeline using Python, Apache Spark, and Slurm to parallelize neural data analysis, enabling efficient large-scale spike train pattern detection.
- **Data Analysis and Visualization**: Provides tools for clustering cell ensembles, visualizing results, and analyzing their correlation with brainwaves, including Sharp-Wave Ripples and Cortical Spindles.

---

## Key Features
1. **High Performance and Scalability**:
   - Apache Spark distributes computations across multiple nodes and CPUs.
   - Ideal for large datasets or high-dimensional spike train analysis.

2. **Fault Tolerance**:
   - Built-in resilience with Spark’s RDDs for handling failures.

3. **Cluster Integration**:
   - Slurm allocates HPC resources, and Spark leverages them efficiently for parallel processing.

4. **In-Memory Processing**:
   - Spark’s in-memory computation optimizes performance for iterative data processing tasks.

---

## File Structure

- **main_assembly_detection_spark.py**:
  Python script that implements the cell assembly detection pipeline using PySpark.

- **submit_job_spark.sh**:
  Slurm job script for running the Spark application on Compute Canada’s cluster.

---

## Dependencies

### Compute Canada Modules
Ensure the following modules are loaded:
```bash
module load java
module load python
module load spark
```

### Python Libraries
Install the required Python libraries in your environment:
```bash
pip install numpy scipy pandas quantities neo elephant pyspark
```

---

## How to Run the Pipeline

### 1. Prepare Data
Ensure your input spike train data is stored in `.zip` files, with one file per task. Files should be named as `{task_id}.zip`.

### 2. Submit the Job
Use the provided Slurm script to submit the Spark job to Compute Canada:

```bash
sbatch submit_job_spark.sh
```

### 3. Monitor the Job
Monitor the job status using Slurm commands:
```bash
squeue -u $USER
```

Spark logs can be accessed in the output and error files specified in `submit_job_spark.sh`.

---

## Detailed Workflow

### 1. **Slurm Job Script**
The Slurm script allocates resources (e.g., memory, CPUs, nodes) for the Spark application:
```bash
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=15:59:59
#SBATCH --output=/home/%u/scratch/peymannr/Results/spark-%u_%A_%a.out
#SBATCH --error=/home/%u/scratch/peymannr/Results/spark-%u_%A_%a.error
#SBATCH --array=17
```
- Adjust the resource allocations as needed.

### 2. **Python Script**
The Python script (`main_assembly_detection_spark.py`) performs the following:
- Reads the input `.zip` files.
- Prepares spike train data using `neo` and `elephant` libraries.
- Distributes the cell assembly detection tasks across Spark’s RDDs.
- Saves the results for each task as a `.pkl` file.

Key Spark commands:
```python
rdd = sc.parallelize(splist)
results = rdd.map(CAD_p).collect()
```

### 3. **Results**
Output files are saved in `.pkl` format, named based on the task ID, rat number, and recording date. Example:
```
CAD_rat1_2023-08-21.pkl
```

---

## Advantages of Using Spark
1. **Parallel Processing**:
   - Tasks are distributed across CPUs and nodes, reducing runtime.

2. **Cluster Integration**:
   - Seamlessly integrates with Compute Canada’s HPC environment via Slurm.

3. **Fault Tolerance**:
   - Automatic recovery of failed tasks.

4. **Scalability**:
   - Efficiently handles large-scale neural data.

---

## Customization
- Adjust **resource allocation** in `submit_job_spark.sh` based on your dataset size and complexity.
- Modify **parameters** in `main_assembly_detection_spark.py` (e.g., bin sizes, lag) to suit your analysis needs.
- Use `spark-submit` options for advanced configurations:
```bash
spark-submit --master yarn --num-executors 4 --executor-memory 16G main_assembly_detection_spark.py
```

---
