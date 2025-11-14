# Concurrent_Fault_Simulator_Group10
A Python + PyQt6 GUI application
Overview
This project implements a concurrent fault simulator for structural Verilog circuits using 5-valued logic (0, 1, X, D, B).
It supports:

Parsing structural Verilog netlists
Reading test vectors from CSV
Topological sorting of circuit elements
Fault-free simulation
Single stuck-at fault generation & collapsing
Multi-vector, multi-timeframe serial fault simulation
Fault coverage reporting
Full PyQt6 GUI for interactive use

Features
✔ Structural Verilog Parsing
Extracts inputs, outputs, wires, and gates
Supports common gate types: AND, OR, NAND, NOR, XOR, XNOR, BUF, NOT

✔ Test Vector Loader
Reads binary vectors from CSV
Automatically validates and aligns with primary inputs

✔ Logic Simulation Engine
Multi-level combinational evaluation
Supports 5-valued logic including D (1/0) and B (0/1)

✔ Fault Model
Detects all Single stuck-at faults (SA0 / SA1)

Fault collapsing (equivalence + dominance)
✔ Serial Fault Simulation
Simulates each fault sequentially over all vectors
Detects first vector and logic level that reveals a fault

Computes fault coverage

✔ Result Generation
Detected & undetected lists
Generates Fault coverage summary
Per-fault detection details
Automatically writes to result file

✔ PyQt6 GUI

Load Verilog file

Load test vectors

Run simulation in background thread

Live console output

Errors and warnings displayed in GUI

Project Structure
fault_simulation_gui.py      # Main PyQt6 GUI application
verilog_parser.py            # Parsing utilities (embedded in full script)
fault_sim_core.py            # Simulation engine (embedded in full script)
test_vectors/                # Sample input CSV files
netlists/                    # Sample Verilog files
results/                     # Generated output files

Installation
Requirements
Python 3.9+

Required libraries:
pip install PyQt6

No external simulators or EDA tools are required.

Usage (GUI)
Run the application:
python fault_simulation_gui.py


In the GUI:

Click Load Verilog File → choose structural netlist
Click Load Test Vector CSV
Press Run Simulation
View:
Console output in GUI
Summary statistics

After completion, a results.txt file is generated with full details.

5-Valued Logic Model (for Sequential Circuits)
Symbol	Meaning
0	Logic 0
1	Logic 1
X	Unknown
D	Faulty 1 (fault-free=1, fault=0)
B	Faulty 0 (fault-free=0, fault=1)

This model enables fault activation + propagation detection.

Stages of the Simulation

Stage 1 – Verilog Parsing

Stage 2 – Test Vector Import

Stage 3 – Topological Sort

Stage 4 – Fault-Free Logic Simulation

Stage 5 – Fault List Generation & Collapsing

Stage 6 – Serial Fault Simulation

Stage 7 – Reporting & Output Generation

Output Files

A result text file is automatically generated containing:

Circuit details

Test vector count

List of detected faults

List of undetected faults

Fault coverage percentage

Time taken for each stage

Example Verilog Format
module c17 (N1, N2, N3, N6, N7, N22, N23);
input N1, N2, N3, N6, N7;
output N22, N23;
wire N10, N11;

nand U1 (N10, N1, N3);
nand U2 (N11, N3, N6);
nand U3 (N22, N11, N7);
nand U4 (N23, N10, N2);
endmodule

Limitations

Only combinational circuits
and sequential elements using D_flip flops.

Future Improvements
To be able to generate Test coverage and fault statistics for all Sequential circuits . (Here only an approach is mentioned using D flip-flops as gates)
To make this program event driven


VCD waveform output

Transition fault model

License

This project is free for educational and research use.
