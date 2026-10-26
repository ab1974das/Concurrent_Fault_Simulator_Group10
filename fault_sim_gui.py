#!/usr/bin/env python3
# fault_sim_gui_eventdriven.py
#
# Event-driven GUI wrapper around the sequential/time-expanded fault simulator
# (based on the prompt_final.docx content). Keeps the same stage outputs and
# appends results to a single Result.txt file per run.
#
# Usage:
#   python fault_sim_gui_eventdriven.py
#
# Requirements:
#   - Python 3.8+
#   - PyQt6

import re
import csv
import copy
import sys
import threading
import traceback
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QFileDialog, QTextEdit, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor

# ----------------------
# STEP 1: PARSE VERILOG
# ----------------------

def parse_verilog(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    statements = [stmt.strip() for stmt in text.split(';') if stmt.strip()]
    inputs, outputs, wires, gates = [], [], [], []
    module_name = None
    gate_types = ['and','or','not','nand','nor','xor','xnor','dff']
    for stmt in statements:
        if stmt.startswith('module'):
            m = re.match(r'module\s+(\w+)\s*\((.*)\)', stmt, flags=re.DOTALL)
            if m:
                module_name = m.group(1)
                port_list = m.group(2).strip()
                for line in port_list.splitlines():
                    line = line.strip().rstrip(',')
                    if not line:
                        continue
                    if line.startswith('input'):
                        nets = line[len('input'):].strip().rstrip(',').split(',')
                        inputs.extend(net.strip() for net in nets if net.strip())
                    elif line.startswith('output'):
                        nets = line[len('output'):].strip().rstrip(',').split(',')
                        outputs.extend(net.strip() for net in nets if net.strip())
    for stmt in statements:
        s = stmt.strip()
        if not s or s.startswith('module') or s == 'endmodule':
            continue
        if s.startswith('input '):
            inputs.extend([x.strip() for x in s[len('input '):].split(',') if x.strip()])
            continue
        if s.startswith('output '):
            outputs.extend([x.strip() for x in s[len('output '):].split(',') if x.strip()])
            continue
        if s.startswith('wire '):
            wires.extend([x.strip() for x in s[len('wire '):].split(',') if x.strip()])
            continue
        for gtype in gate_types:
            if s.lower().startswith(gtype):
                rest = s[len(gtype):].strip()
                if rest.startswith('('):
                    m_conn = re.match(r'\(([^)]*)\)', rest)
                    if not m_conn:
                        continue
                    nets = [n.strip() for n in m_conn.group(1).split(',') if n.strip()]
                    if nets:
                        gates.append({'type': gtype, 'output': nets[0], 'inputs': nets[1:]})
                else:
                    instances = re.findall(r'(\w+)\s*\(([^)]*)\)', rest)
                    for _, args in instances:
                        nets = [n.strip() for n in args.split(',') if n.strip()]
                        if nets:
                            gates.append({'type': gtype, 'output': nets[0], 'inputs': nets[1:]})
                break
    inputs, outputs, wires = map(lambda x: list(dict.fromkeys(x)), [inputs, outputs, wires])
    print("\nSTAGE 1: VERILOG PARSING RESULTS")
    print(f"Module Name: {module_name}")
    print(f"Primary Inputs  ({len(inputs)}): {inputs}")
    print(f"Primary Outputs ({len(outputs)}): {outputs}")
    print(f"Internal Wires  ({len(wires)}): {wires}")
    print(f"Total Gates: {len(gates)}")
    for g in gates:
        print(f"  Gate: {g['type'].upper():5s} | Output: {g['output']} | Inputs: {g['inputs']}")
    return {'module': module_name, 'inputs': inputs, 'outputs': outputs, 'wires': wires, 'gates': gates}

# ----------------------
# STEP 2: READ VECTORS
# ----------------------

def read_test_vectors(csv_path, primary_inputs):
    vectors = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        header = [h.strip() for h in header]
        if sorted(header) != sorted(primary_inputs):
            raise ValueError(f"CSV header {header} does not match expected inputs {primary_inputs}")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Incorrect number of columns: {row}")
            vector = {name: int(val) for name, val in zip(header, row)}
            if any(v not in (0, 1) for v in vector.values()):
                raise ValueError(f"Non-binary value found in row: {row}")
            vectors.append(vector)
    print("\nSTAGE 2: TEST VECTOR FILE ANALYSIS")
    print(f"Total Vectors Read: {len(vectors)}")
    print("First few vectors:")
    for v in vectors[:3]:
        print(" ", v)
    return vectors

# ----------------------
# STEP 3: TOPOGRAPHICAL SORT
# ----------------------

def topo_sort_gates(circuit):
    inputs = set(circuit['inputs'])
    gates = circuit['gates']
    driver = {g['output']: g for g in gates}

    def get_level(gate, visiting=None):
        if visiting is None:
            visiting = set()
        if 'level' in gate:
            return gate['level']
        if gate['output'] in visiting:
            raise ValueError(f"Combinational cycle detected involving net '{gate['output']}'. Ensure DFFs separate combinational loops.")
        visiting.add(gate['output'])
        level = 0
        for inp in gate['inputs']:
            if inp in inputs:
                level = max(level, 0)
            elif inp in driver and driver[inp]['type'].lower() == 'dff':
                level = max(level, 0)
            elif inp in driver:
                level = max(level, get_level(driver[inp], visiting))
        visiting.remove(gate['output'])
        gate['level'] = level + 1
        return gate['level']

    for g in gates:
        get_level(g)

    sorted_gates = sorted(gates, key=lambda x: x['level'])
    print("\nSTAGE 3: TOPOLOGICAL ORDER OF GATES")
    for g in sorted_gates:
        print(f"Level {g['level']}: {g['type'].upper():5s} | Output: {g['output']:<6s} | Inputs: {g['inputs']}")
    return sorted_gates

# ----------------------
# STEP 4: EVAL + SIM (sequential)
# ----------------------

def eval_gate(gtype, input_vals):
    g = gtype.lower()
    if g == 'and':
        return int(all(input_vals))
    if g == 'or':
        return int(any(input_vals))
    if g == 'not':
        return int(not input_vals[0])
    if g == 'nand':
        return int(not all(input_vals))
    if g == 'nor':
        return int(not any(input_vals))
    if g == 'xor':
        res = 0
        for v in input_vals:
            res ^= (v & 1)
        return int(res)
    if g == 'xnor':
        res = 0
        for v in input_vals:
            res ^= (v & 1)
        return int(not res)
    raise ValueError(f"Unsupported gate type: {gtype}")

def simulate_fault_free(circuit, sorted_gates, test_vectors, dff_init_states=None, verbose=True):
    dff_gates = [g for g in circuit['gates'] if g['type'].lower() == 'dff']
    dff_outputs = [g['output'] for g in dff_gates]
    dff_states = {}
    if dff_init_states:
        dff_states.update(dff_init_states)
    for dout in dff_outputs:
        dff_states.setdefault(dout, 0)
    results = []

    # helper: infer D input for dff (avoid picking clk)
    def infer_d_input(in_nets, net_vals):
        clk_names = {'clk','clock','clkin'}
        for n in in_nets:
            if n.lower() not in clk_names:
                return int(net_vals.get(n, 0))
        # fallback
        return int(net_vals.get(in_nets[-1], 0)) if in_nets else 0

    for vec_idx, vec in enumerate(test_vectors, start=1):
        net_vals = {}
        # primary inputs
        for pin in circuit['inputs']:
            net_vals[pin] = int(vec[pin])
        # place current dff states on outputs
        for dout in dff_outputs:
            net_vals[dout] = int(dff_states.get(dout, 0))
        # evaluate gates in topo order
        for g in sorted_gates:
            gtype = g['type'].lower()
            out_net = g['output']
            in_nets = g['inputs']
            in_vals = [net_vals.get(n, 0) for n in in_nets]
            if gtype == 'dff':
                d_val = infer_d_input(in_nets, net_vals)
                net_vals.setdefault(f"{out_net}_D_input", d_val)
                continue
            try:
                out_val = eval_gate(gtype, in_vals)
            except Exception as e:
                print(f"Warning: eval problem for {gtype} -> {e}. Defaulting {out_net} to 0.")
                out_val = 0
            net_vals[out_net] = int(out_val)
        # update dff states at end of frame (clock edge)
        for dout in dff_outputs:
            dff_states[dout] = int(net_vals.get(f"{dout}_D_input", dff_states.get(dout, 0)))
        results.append(net_vals)
        if verbose:
            print("\nSTAGE 4: FAULT-FREE SIMULATION — Vector #{}".format(vec_idx))
            print("Primary Inputs:")
            for p in circuit['inputs']:
                print(f"  {p} = {net_vals.get(p, 0)}")
            print("Primary Outputs:")
            for po in circuit['outputs']:
                print(f"  {po} = {net_vals.get(po, 0)}")
            if circuit['wires']:
                print("Some internal wires (sample):")
                for w in circuit['wires'][:20]:
                    print(f"  {w} = {net_vals.get(w, 0)}")
    return results, dff_states

# ----------------------
# STEP 5: FAULT LIST + COLLAPSE
# ----------------------

def generate_full_fault_list(circuit, exclude_clock=True):
    nets = circuit['inputs'] + circuit['wires'] + circuit['outputs']
    if exclude_clock:
        def is_clock(n):
            nlow = n.lower()
            return ('clk' in nlow) or ('clock' in nlow) or ('clkin' in nlow)
        nets = [n for n in nets if not is_clock(n)]
    faults = [f"{n}/SA0" for n in nets] + [f"{n}/SA1" for n in nets]
    print("\nSTAGE 5A: FULL FAULT LIST (clock-nets excluded)" if exclude_clock else "\nSTAGE 5A: FULL FAULT LIST")
    print(f"Total Nets: {len(nets)}")
    print(f"Total Faults (Full List): {len(faults)}")
    print("Sample (first 20):", faults[:20])
    return faults

def collapse_faults_equivalence(circuit, full_faults):
    fault_set = set(full_faults)
    removed = set()
    eq_map = []
    def sa(n,v): return f"{n}/SA{v}"
    for g in circuit['gates']:
        t = g['type'].lower()
        ins, y = g['inputs'], g['output']
        if t == 'and':
            group = [sa(a,0) for a in ins if sa(a,0) in fault_set]
            if sa(y,0) in fault_set: group.append(sa(y,0))
            if len(group) > 1:
                keep = group[0]; group.remove(keep)
                for f in group: removed.add(f)
                eq_map.append((keep, group))
        elif t == 'or':
            group = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if sa(y,1) in fault_set: group.append(sa(y,1))
            if len(group) > 1:
                keep = group[0]; group.remove(keep)
                for f in group: removed.add(f)
                eq_map.append((keep, group))
        elif t == 'nand':
            group = [sa(a,0) for a in ins if sa(a,0) in fault_set]
            if sa(y,1) in fault_set: group.append(sa(y,1))
            if len(group) > 1:
                keep = group[0]; group.remove(keep)
                for f in group: removed.add(f)
                eq_map.append((keep, group))
        elif t == 'nor':
            group = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if sa(y,0) in fault_set: group.append(sa(y,0))
            if len(group) > 1:
                keep = group[0]; group.remove(keep)
                for f in group: removed.add(f)
                eq_map.append((keep, group))
        elif t == 'not' and ins:
            a = ins[0]
            if sa(a,0) in fault_set and sa(y,1) in fault_set:
                removed.add(sa(y,1))
                eq_map.append((sa(a,0), [sa(y,1)]))
            if sa(a,1) in fault_set and sa(y,0) in fault_set:
                removed.add(sa(y,0))
                eq_map.append((sa(a,1), [sa(y,0)]))
    collapsed_faults = sorted(f for f in fault_set if f not in removed)
    print("\nSTAGE 5B: EQUIVALENCE COLLAPSING")
    print(f"Before collapsing: {len(full_faults)} faults")
    print(f"After collapsing : {len(collapsed_faults)} faults")
    print(f"Faults removed    : {len(removed)}")
    for k,grp in eq_map[:10]:
        print(f"  {k} ≡ {grp}")
    return collapsed_faults, eq_map

def collapse_faults_dominance(circuit, eq_faults):
    fault_set = set(eq_faults)
    removed = set()
    dom_map = []
    def sa(n,v): return f"{n}/SA{v}"
    # NOTE: a dominating fault (e.g. a gate-output SA1 on an AND gate) may
    # only be dropped if at least one of its dominating "witness" faults
    # (e.g. an input SA1 on that same gate) is still present in the fault
    # set under its own name. If equivalence collapsing already folded that
    # witness into some other net's fault, there is nothing left in the
    # simulated list whose detection is guaranteed to imply detection of
    # the output fault -- so it must NOT be removed in that case.
    for g in circuit['gates']:
        t = g['type'].lower()
        ins, y = g['inputs'], g['output']
        if t == 'and' and sa(y,1) in fault_set:
            witnesses = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if witnesses:
                dom_map.append((sa(y,1), witnesses))
                removed.add(sa(y,1))
        elif t == 'nand' and sa(y,0) in fault_set:
            witnesses = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if witnesses:
                dom_map.append((sa(y,0), witnesses))
                removed.add(sa(y,0))
        elif t == 'or' and sa(y,0) in fault_set:
            witnesses = [sa(a,0) for a in ins if sa(a,0) in fault_set]
            if witnesses:
                dom_map.append((sa(y,0), witnesses))
                removed.add(sa(y,0))
        elif t == 'nor' and sa(y,1) in fault_set:
            witnesses = [sa(a,0) for a in ins if sa(a,0) in fault_set]
            if witnesses:
                dom_map.append((sa(y,1), witnesses))
                removed.add(sa(y,1))
        elif t == 'xor' and sa(y,1) in fault_set:
            witnesses = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if witnesses:
                dom_map.append((sa(y,1), witnesses))
                removed.add(sa(y,1))
        elif t == 'xnor' and sa(y,0) in fault_set:
            witnesses = [sa(a,1) for a in ins if sa(a,1) in fault_set]
            if witnesses:
                dom_map.append((sa(y,0), witnesses))
                removed.add(sa(y,0))
    collapsed_final = sorted(f for f in fault_set if f not in removed)
    print("\nSTAGE 5C: DOMINANCE COLLAPSING")
    print(f"Before dominance: {len(eq_faults)} faults")
    print(f"After dominance : {len(collapsed_final)} faults")
    print(f"Faults removed  : {len(removed)}")
    for d,subs in dom_map[:10]:
        if subs:
            print(f"  {d} → {subs}")
    return collapsed_final, dom_map

def generate_and_collapse_faults(circuit):
    full = generate_full_fault_list(circuit, exclude_clock=True)
    eq_collapsed, eq_map = collapse_faults_equivalence(circuit, full)
    dom_collapsed, dom_map = collapse_faults_dominance(circuit, eq_collapsed)
    print("\nSTAGE 5D: FAULT LIST SUMMARY")
    print(f"Full fault count       : {len(full)}")
    print(f"After equivalence      : {len(eq_collapsed)}")
    print(f"After dominance        : {len(dom_collapsed)}")
    print(f"Faults that will actually be simulated in Stage 6: {len(dom_collapsed)} "
          f"(reduced from {len(full)} full faults, "
          f"{100.0 * (len(full) - len(dom_collapsed)) / len(full):.1f}% reduction)" if full else "")
    return {'full': full, 'eq_collapsed': eq_collapsed, 'dom_collapsed': dom_collapsed, 'eq_map': eq_map, 'dom_map': dom_map}

# ----------------------
# STEP 6b: EXPAND COLLAPSED-LIST RESULTS BACK TO THE FULL FAULT LIST
# ----------------------

def expand_coverage_to_full_list(faults_info, detection_map):
    """
    Map Stage 6 detection results (computed only on the collapsed fault
    list) back onto the FULL, uncollapsed fault list, so a full-list
    coverage number can be reported alongside the standard collapsed-fault
    coverage.

    - Equivalence-collapsed faults are PROVABLY co-detected with their
      representative, in both directions (that's what equivalence means).
      So each removed fault inherits its representative's detected /
      undetected status exactly -- no approximation here.

    - Dominance-collapsed faults are only guaranteed in ONE direction: any
      test that detects the dominated "witness" fault is guaranteed to
      also detect the dominating fault that was removed. The converse is
      NOT guaranteed -- if the witness wasn't detected by the applied
      vectors, the dominating fault might still be detectable by some
      other vector we didn't simulate, but we can't claim that without
      actually simulating it. So here it is conservatively marked
      undetected. This makes the expanded full-list number a safe LOWER
      BOUND on true full-list coverage, not necessarily an exact count --
      the same convention ATPG tools use when back-annotating collapsed
      results.
    """
    # Equivalence collapsing is a single forward sweep over the gate list
    # (see collapse_faults_equivalence), not a transitive closure -- so a
    # fault that is used as a "representative" for one grouping can itself
    # get folded into a DIFFERENT representative by a later grouping in the
    # same pass (e.g. a NOT gate's fault keeps some net as representative,
    # but an AND/OR gate downstream later folds that same net's fault into
    # yet another representative). A naive single pass over eq_map in
    # insertion order resolves these chains incorrectly. Union-Find over
    # every fault name that appears anywhere in eq_map gives the correct
    # equivalence classes regardless of chain order.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rep, folded in faults_info['eq_map']:
        for f in folded:
            union(rep, f)

    # Ground truth: the faults actually simulated (dom_collapsed).
    known_detected = set(f for f, d in detection_map.items() if d['detected'])
    known_undetected = set(f for f, d in detection_map.items() if not d['detected'])

    # Dominance: removed (dominating) fault <- [dominated witness fault(s)].
    # Guaranteed only in the direction "witness detected => dominating
    # fault detected" (see collapse_faults_dominance docstring/comments).
    # If no witness was detected, conservatively mark undetected -- this is
    # what makes the expanded number a safe LOWER BOUND rather than exact.
    for removed, witnesses in faults_info['dom_map']:
        if any(w in known_detected for w in witnesses):
            known_detected.add(removed)
        else:
            known_undetected.add(removed)

    # An equivalence class is DETECTED if ANY member's status is known and
    # detected (equivalent faults are guaranteed to be co-detected, so one
    # detected member proves the whole class is); otherwise UNDETECTED if
    # any member's status is known and undetected.
    class_detected = {}
    for f in known_detected:
        class_detected[find(f)] = True
    for f in known_undetected:
        class_detected.setdefault(find(f), False)

    full_detected = set()
    full_undetected = set()
    for f in faults_info['full']:
        (full_detected if class_detected.get(find(f), False) else full_undetected).add(f)

    return full_detected, full_undetected

# ----------------------
# STEP 6: TIME-EXPANDED SEQUENTIAL FAULT SIM (5-valued)
# ----------------------
def fault_simulation_concurrent(fault_list_to_simulate, circuit, sorted_gates,
                                fault_free_results_binary, test_vectors,
                                verbose_sample=10):
    """
    Concurrent stuck-at fault simulation using per-net concurrent fault lists.

    IMPORTANT: fault_list_to_simulate should be the COLLAPSED fault list
    (faults_info['dom_collapsed']) rather than the full fault list. Every
    fault that equivalence/dominance collapsing removed is guaranteed to be
    detected by exactly the same set of test vectors as the representative
    fault it collapsed into, so simulating the full list wastes time without
    changing the result set (once you fold the collapsed faults back in for
    reporting, if you need per-fault statistics for the full list).

    For each test vector:
        1. Simulate the good circuit once.
        2. Maintain a concurrent fault list at every net.
        3. Propagate all active faults through the circuit.
        4. Detect faults appearing at primary outputs.
        5. Update faulty DFF states for the next vector.
    """

    VAL0, VAL1, VALX, VALD, VALB = '0', '1', 'X', 'D', 'B'

    def parse_fault(fstr):
        net, sa = fstr.split('/SA')
        return net, int(sa)

    # ------------------------------------------------------------
    # 5-valued gate evaluation
    # D = good 1, faulty 0
    # B = good 0, faulty 1
    # ------------------------------------------------------------
    def eval_gate_5val(gtype, vals):
        g = gtype.lower()

        if g == 'not':
            return {
                VAL0: VAL1,
                VAL1: VAL0,
                VALD: VALB,
                VALB: VALD
            }.get(vals[0], VALX)

        if any(v == VALX for v in vals):
            return VALX

        good = []
        faulty = []

        for v in vals:
            if v == VALD:
                good.append(1)
                faulty.append(0)

            elif v == VALB:
                good.append(0)
                faulty.append(1)

            else:
                b = 1 if v == VAL1 else 0
                good.append(b)
                faulty.append(b)

        def evaluate(bits):
            if g == 'and':
                return int(all(bits))

            if g == 'or':
                return int(any(bits))

            if g == 'nand':
                return int(not all(bits))

            if g == 'nor':
                return int(not any(bits))

            if g == 'xor':
                result = 0
                for b in bits:
                    result ^= b
                return result

            if g == 'xnor':
                result = 0
                for b in bits:
                    result ^= b
                return int(not result)

            return None

        good_value = evaluate(good)
        faulty_value = evaluate(faulty)

        if good_value == faulty_value:
            return VAL1 if good_value else VAL0

        if good_value == 1 and faulty_value == 0:
            return VALD

        return VALB

    # ------------------------------------------------------------
    # DFF information
    # ------------------------------------------------------------
    dff_gates = [
        g for g in circuit['gates']
        if g['type'].lower() == 'dff'
    ]

    dff_outputs = [g['output'] for g in dff_gates]

    # Good DFF states
    good_dff_states = {
        q: 0 for q in dff_outputs
    }

    # Concurrent fault lists stored in DFF states
    #
    # Example:
    # dff_fault_states['Q'] =
    # {
    #     'A/SA0': 'D',
    #     'Q/SA1': 'B'
    # }
    #
    dff_fault_states = {
        q: {} for q in dff_outputs
    }

    fault_set = set(fault_list_to_simulate)

    # ------------------------------------------------------------
    # Detection information
    # ------------------------------------------------------------
    detection_map = {
        fault: {
            'detected': False,
            'by_vector': None,
            'first_level': None
        }
        for fault in fault_list_to_simulate
    }

    detected_faults = set()

    total_faults = len(fault_list_to_simulate)
    total_vectors = len(test_vectors)

    print("\nSTAGE 6: CONCURRENT FAULT SIMULATION (5-valued, on COLLAPSED fault list)")
    print(f"Total faults to simulate: {total_faults}")
    print(f"Total time frames       : {total_vectors}")

    # ============================================================
    # VECTOR LOOP
    #
    # This is the important change.
    #
    # V1 -> all faults
    # V2 -> all remaining faults
    # V3 -> all remaining faults
    # ...
    # ============================================================
    for vector_number, vector in enumerate(test_vectors, start=1):

        # ========================================================
        # 1. SIMULATE GOOD CIRCUIT
        # ========================================================
        good = {}

        # Primary inputs
        for pin in circuit['inputs']:
            good[pin] = int(vector.get(pin, 0))

        # Current good DFF states
        for q in dff_outputs:
            good[q] = good_dff_states[q]

        # Evaluate circuit in topological order
        for gate in sorted_gates:

            gate_type = gate['type'].lower()
            output = gate['output']
            inputs = gate['inputs']

            # DFF
            if gate_type == 'dff':

                d_net = next(
                    (
                        n for n in inputs
                        if n.lower()
                        not in {'clk', 'clock', 'clkin'}
                    ),
                    inputs[-1] if inputs else None
                )

                if d_net is not None:
                    good[f"{output}_D_input"] = \
                        good.get(d_net, 0)

                continue

            input_values = [
                good.get(n, 0)
                for n in inputs
            ]

            good[output] = eval_gate(
                gate_type,
                input_values
            )

        # ========================================================
        # 2. CREATE CONCURRENT FAULT LISTS
        #
        # fault_lists[net] =
        #       {fault : faulty_value}
        #
        # Only faults currently affecting a net are stored.
        # ========================================================
        fault_lists = {}

        # --------------------------------------------------------
        # Primary-input faults
        # --------------------------------------------------------
        for pin in circuit['inputs']:

            good_value = good[pin]

            for sa_value in [0, 1]:

                fault = f"{pin}/SA{sa_value}"

                if fault not in fault_set:
                    continue

                # If stuck value is same as good value,
                # the fault is currently masked.
                if sa_value == good_value:
                    continue

                # Convert to D/B notation.
                #
                # good=1, faulty=0 -> D
                # good=0, faulty=1 -> B
                faulty_symbol = (
                    VALD if good_value == 1 else VALB
                )

                fault_lists.setdefault(pin, {})[
                    fault
                ] = faulty_symbol

        # --------------------------------------------------------
        # Faulty DFF states enter the current vector
        # --------------------------------------------------------
        for q in dff_outputs:

            if dff_fault_states[q]:
                fault_lists[q] = dict(
                    dff_fault_states[q]
                )

            # Stuck-at faults physically located on Q
            q_sa0 = f"{q}/SA0"
            q_sa1 = f"{q}/SA1"

            if (
                q_sa0 in fault_set
                and good[q] != 0
            ):
                fault_lists.setdefault(q, {})[
                    q_sa0
                ] = VALD

            if (
                q_sa1 in fault_set
                and good[q] != 1
            ):
                fault_lists.setdefault(q, {})[
                    q_sa1
                ] = VALB

        # ========================================================
        # 3. CONCURRENT FAULT PROPAGATION
        #
        # Each gate receives:
        #
        #   good input values
        #   + concurrent fault list
        #
        # All active faults are propagated at the same time
        # conceptually through the gate.
        # ========================================================
        for gate in sorted_gates:

            gate_type = gate['type'].lower()
            output = gate['output']
            inputs = gate['inputs']

            # ----------------------------------------------------
            # DFF
            # ----------------------------------------------------
            if gate_type == 'dff':

                d_net = next(
                    (
                        n for n in inputs
                        if n.lower()
                        not in {'clk', 'clock', 'clkin'}
                    ),
                    inputs[-1] if inputs else None
                )

                if d_net is not None:

                    # Faults on D become faults on Q
                    # in the next time frame.
                    fault_lists[
                        f"{output}_D_input"
                    ] = dict(
                        fault_lists.get(d_net, {})
                    )

                continue

            # ----------------------------------------------------
            # Find all faults currently affecting any input
            # ----------------------------------------------------
            active_faults = set()

            for net in inputs:
                active_faults.update(
                    fault_lists.get(net, {}).keys()
                )

            # ----------------------------------------------------
            # A fault located at the gate output is introduced here
            # ----------------------------------------------------
            output_sa0 = f"{output}/SA0"
            output_sa1 = f"{output}/SA1"

            if output_sa0 in fault_set:
                active_faults.add(output_sa0)

            if output_sa1 in fault_set:
                active_faults.add(output_sa1)

            if not active_faults:
                continue

            output_fault_list = {}

            good_output = good.get(output, 0)

            # ----------------------------------------------------
            # Propagate each entry of the concurrent fault list
            # ----------------------------------------------------
            for fault in active_faults:

                fault_net, stuck_value = parse_fault(fault)

                stuck_symbol = (
                    VAL1 if stuck_value == 1
                    else VAL0
                )

                faulty_inputs = []

                for net in inputs:

                    # If this particular fault is present
                    # on the input, use its faulty value.
                    if fault in fault_lists.get(net, {}):

                        faulty_inputs.append(
                            fault_lists[net][fault]
                        )

                    else:

                        # Otherwise use the fault-free value.
                        faulty_inputs.append(
                            VAL1 if good.get(net, 0)
                            else VAL0
                        )

                # Evaluate faulty circuit
                faulty_output = eval_gate_5val(
                    gate_type,
                    faulty_inputs
                )

                # ------------------------------------------------
                # Fault physically located at gate output
                # ------------------------------------------------
                if fault_net == output:
                    faulty_output = stuck_symbol

                    # Convert the output to D/B representation
                    if faulty_output != (
                        VAL1 if good_output else VAL0
                    ):
                        faulty_output = (
                            VALD
                            if good_output == 1
                            else VALB
                        )

                # ------------------------------------------------
                # Keep fault only if it differs from good circuit
                # ------------------------------------------------
                if faulty_output == VALX:

                    output_fault_list[fault] = VALX

                else:

                    if faulty_output == VALD:
                        faulty_bit = 0

                    elif faulty_output == VALB:
                        faulty_bit = 1

                    else:
                        faulty_bit = (
                            1 if faulty_output == VAL1
                            else 0
                        )

                    # Fault is propagated only if observable at
                    # this point.
                    if faulty_bit != good_output:

                        output_fault_list[fault] = (
                            VALD
                            if good_output == 1
                            else VALB
                        )

            # ----------------------------------------------------
            # Update concurrent fault list of this gate
            # ----------------------------------------------------
            if output_fault_list:
                fault_lists[output] = output_fault_list
            else:
                fault_lists.pop(output, None)

        # ========================================================
        # 4. FAULT DETECTION
        #
        # If a fault reaches a primary output as D/B, it is detected.
        # ========================================================
        for primary_output in circuit['outputs']:

            output_faults = fault_lists.get(
                primary_output,
                {}
            )

            for fault, value in output_faults.items():

                if fault in detected_faults:
                    continue

                if value in (VALD, VALB):

                    detected_faults.add(fault)

                    detection_map[fault] = {
                        'detected': True,
                        'by_vector': vector_number,
                        'first_level': next(
                            (
                                g.get('level')
                                for g in sorted_gates
                                if g['output']
                                == primary_output
                            ),
                            None
                        )
                    }

        # ========================================================
        # 5. UPDATE DFF STATES
        # ========================================================
        next_good_states = {}
        next_fault_states = {}

        for gate in dff_gates:

            q = gate['output']

            # Good Q for next vector
            next_good_states[q] = int(
                good.get(
                    f"{q}_D_input",
                    good_dff_states[q]
                )
            )

            # Faults propagated from D to Q
            next_fault_states[q] = dict(
                fault_lists.get(
                    f"{q}_D_input",
                    {}
                )
            )

            # A stuck-at fault on Q persists
            q_sa0 = f"{q}/SA0"
            q_sa1 = f"{q}/SA1"

            if (
                q_sa0 in fault_set
                and next_good_states[q] != 0
            ):
                next_fault_states[q][q_sa0] = VALD

            if (
                q_sa1 in fault_set
                and next_good_states[q] != 1
            ):
                next_fault_states[q][q_sa1] = VALB

        good_dff_states = next_good_states
        dff_fault_states = next_fault_states

        print(
            f"Processed vector {vector_number}/{total_vectors}; "
            f"faults detected so far: "
            f"{len(detected_faults)}"
        )

    # ============================================================
    # FINAL STATISTICS
    # ============================================================
    total_detected = len(detected_faults)

    detection_percentage = (
        100.0 * total_detected / total_faults
        if total_faults else 0.0
    )

    print(
        "\nSTAGE 6 CONCURRENT FAULT SIMULATION SUMMARY"
    )

    print(
        f"Total faults simulated (collapsed list) : "
        f"{total_faults}"
    )

    print(
        f"Detected faults        : "
        f"{total_detected}"
    )

    print(
        f"Detection percentage   : "
        f"{detection_percentage:.2f}%"
    )

    print(
        "\nSample detected faults "
        "(fault -> frame, first_change_level):"
    )

    for fault in list(detected_faults)[:verbose_sample]:

        info = detection_map[fault]

        print(
            f"  {fault} -> frame "
            f"#{info['by_vector']}, "
            f"first_level={info['first_level']}"
        )

    undetected = [
        f for f in fault_list_to_simulate
        if f not in detected_faults
    ]

    if undetected:

        print("\nSample undetected faults:")

        for fault in undetected[:verbose_sample]:
            print(f"  {fault}")

    return detection_map

# ----------------------
# STEP 7: WRITE RESULTS
# ----------------------

def write_results_to_file(circuit, faults_info, detection_map, sorted_gates, filename="Result.txt",
                           full_detected=None, full_undetected=None):
    # Append a timestamped run block so results from multiple GUI runs stack in the same file.
    with open(filename, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Run Timestamp: {datetime.now().isoformat()}\n")
        f.write("="*80 + "\n\n")
        f.write("VLSI FAULT SIMULATION RESULTS:\n\n")
        f.write("MODULE INFORMATION\n")
        f.write(f"Module Name: {circuit['module']}\n")
        f.write(f"Primary Inputs  : {circuit['inputs']}\n")
        f.write(f"Primary Outputs : {circuit['outputs']}\n\n")
        f.write("TOPOLOGICAL ORDER OF GATES\n")
        for g in sorted_gates:
            f.write(f"Level {g['level']}: {g['type'].upper():5s} | Output: {g['output']:<6s} | Inputs: {g['inputs']}\n")
        f.write("\n")
        f.write("FAULT STATISTICS\n")
        f.write(f"Total faults (full list)      : {len(faults_info['full'])}\n")
        f.write("Sample faults (first 20):" + str(faults_info['full'][:20]) + "\n")
        f.write(f"After equivalence collapsing  : {len(faults_info['eq_collapsed'])}\n")
        f.write(f"After dominance collapsing    : {len(faults_info['dom_collapsed'])}\n")
        f.write(f"Sample faults after dominance collapsing    : {faults_info['dom_collapsed'][:20]}\n")
        if faults_info['full']:
            reduction = 100.0 * (len(faults_info['full']) - len(faults_info['dom_collapsed'])) / len(faults_info['full'])
            f.write(f"Simulation workload reduction from collapsing: {reduction:.1f}% "
                    f"({len(faults_info['full'])} -> {len(faults_info['dom_collapsed'])} faults actually simulated)\n\n")
        else:
            f.write("\n")
        detected = [f for f, d in detection_map.items() if d['detected']]
        undetected = [f for f, d in detection_map.items() if not d['detected']]
        f.write("FAULT SIMULATION SUMMARY (on collapsed fault list)\n")
        f.write(f"Detected faults   : {len(detected)}\n")
        f.write(f"Undetected faults : {len(undetected)}\n")
        f.write(f"Detection Rate    : {100.0 * len(detected)/len(detection_map):.2f}%\n\n")
        if full_detected is not None and full_undetected is not None:
            full_total = len(full_detected) + len(full_undetected)
            f.write("FULL-LIST FAULT COVERAGE (back-annotated from collapsed results)\n")
            f.write(f"Detected (full list)   : {len(full_detected)}/{full_total}\n")
            f.write(f"Full-list coverage      : {100.0*len(full_detected)/full_total:.2f}% "
                    f"(lower bound where dominance links are unresolved)\n\n")
        f.write("SAMPLE DETECTED FAULTS:\n")
        for f_name in detected[:10]:
            info = detection_map[f_name]
            f.write(f"  {f_name} -> frame #{info['by_vector']}, first_change_level={info['first_level']}\n")
        f.write("\nSAMPLE UNDETECTED FAULTS:\n")
        for f_name in undetected[:10]:
            f.write(f"  {f_name}\n")
        f.write("\nEND OF REPORT\n\n")
    print(f"\nResults have been written/appended to '{filename}'.")

# ----------------------
# GUI: Event-driven wrapper using PyQt6
# ----------------------

class StreamEmitter(QObject):
    new_text = pyqtSignal(str)

class FaultSimulatorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLSI Fault Simulation Tool (Event-driven)")
        self.setGeometry(100, 100, 1000, 700)
        self.stop_requested = False
        self.running = False

        # UI elements
        self.label_verilog = QLabel("No Verilog file selected")
        self.label_csv = QLabel("No CSV file selected")
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)

        btn_load_verilog = QPushButton("Load Verilog File")
        btn_load_csv = QPushButton("Load Test Vectors (CSV)")
        btn_run = QPushButton("Run Simulation")
        btn_stop = QPushButton("Stop")

        btn_load_verilog.clicked.connect(self.load_verilog)
        btn_load_csv.clicked.connect(self.load_csv)
        btn_run.clicked.connect(self.run_simulation)
        btn_stop.clicked.connect(self.stop_simulation)

        layout = QVBoxLayout()
        header = QLabel("<b>Fault Simulation GUI (Event-driven)</b>")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        layout.addWidget(self.label_verilog)
        layout.addWidget(self.label_csv)

        hbox = QHBoxLayout()
        hbox.addWidget(btn_load_verilog)
        hbox.addWidget(btn_load_csv)
        layout.addLayout(hbox)

        layout.addWidget(btn_run)
        layout.addWidget(btn_stop)
        layout.addWidget(QLabel("Console Output:"))
        layout.addWidget(self.text_output)
        self.setLayout(layout)

        # Redirect stdout to GUI
        self.emitter = StreamEmitter()
        self.emitter.new_text.connect(self.append_output)
        sys.stdout = self

        self.verilog_path = None
        self.csv_path = None
        self.sim_thread = None

    # sys.stdout replacement methods
    def write(self, text):
        # emit text to GUI (keeps original prints intact)
        self.emitter.new_text.emit(str(text))

    def flush(self):
        pass

    def append_output(self, text):
        # append to QTextEdit
        self.text_output.moveCursor(QTextCursor.MoveOperation.End)
        self.text_output.insertPlainText(text)
        self.text_output.moveCursor(QTextCursor.MoveOperation.End)

    def load_verilog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Verilog File", "", "Verilog Files (*.v *.sv *.vh);;All Files (*)")
        if file_path:
            self.verilog_path = file_path
            self.label_verilog.setText(f"Verilog File: {file_path}")

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.csv_path = file_path
            self.label_csv.setText(f"CSV File: {file_path}")

    def stop_simulation(self):
        # Set stop_requested and, if simulation is running, wait until thread finishes then quit app
        if self.running:
            self.stop_requested = True
            QMessageBox.information(self, "Stop Requested", "Stop requested. Current run will finish quickly and program will exit afterward.")
        else:
            # not running -> just quit
            QApplication.instance().quit()

    def run_simulation(self):
        if not self.verilog_path or not self.csv_path:
            QMessageBox.warning(self, "Missing Files", "Please select both Verilog and CSV files before running.")
            return

        if self.running:
            QMessageBox.information(self, "Already Running", "A simulation is already running. Wait for it to finish or click Stop.")
            return

        self.stop_requested = False
        self.sim_thread = threading.Thread(target=self.simulation_flow, daemon=True)
        self.sim_thread.start()

    def simulation_flow(self):
        # Mark as running
        self.running = True
        try:
            print("\n==============================")
            print("NEW SIMULATION SESSION STARTED")
            print("==============================\n")

            # Stage 1
            circuit = parse_verilog(self.verilog_path)

            # Stage 2
            test_vectors = read_test_vectors(self.csv_path, circuit['inputs'])

            # Stage 3
            sorted_gates = topo_sort_gates(circuit)

            # Stage 4
            sim_results, dff_states = simulate_fault_free(circuit, sorted_gates, test_vectors, dff_init_states=None, verbose=True)

            # Stage 5
            faults_info = generate_and_collapse_faults(circuit)

            # ---- FIX: simulate the COLLAPSED fault list, not the full one ----
            # This is the whole point of doing fault collapsing: every fault
            # removed by equivalence/dominance collapsing is detected by the
            # same vectors as the representative fault it collapsed into, so
            # only the survivors (faults_info['dom_collapsed']) need to be
            # explicitly propagated through the circuit in Stage 6.
            faults_to_simulate = faults_info['dom_collapsed']

            # Stage 6
            detection_map = fault_simulation_concurrent(faults_to_simulate, circuit, sorted_gates, sim_results, test_vectors)

            # Stage 6b - expand collapsed-list results back onto the full fault list
            full_detected, full_undetected = expand_coverage_to_full_list(faults_info, detection_map)
            full_total = len(full_detected) + len(full_undetected)
            print("\nSTAGE 6b: FULL-LIST FAULT COVERAGE (back-annotated from collapsed results)")
            print(f"Collapsed fault coverage : {100.0*len([f for f,d in detection_map.items() if d['detected']])/len(detection_map):.2f}% "
                  f"({len(detection_map)} faults simulated)")
            print(f"Full-list fault coverage : {100.0*len(full_detected)/full_total:.2f}% "
                  f"({len(full_detected)}/{full_total} of all {len(faults_info['full'])} original faults; "
                  f"lower bound where dominance links are unresolved)")

            # Stage 7 - append to Result.txt
            write_results_to_file(circuit, faults_info, detection_map, sorted_gates, filename="Result.txt",
                                   full_detected=full_detected, full_undetected=full_undetected)

            print("\nSimulation complete. Results appended to 'Result.txt'.\n")

            # After run: if stop requested, exit app; otherwise stay ready for next run
            if self.stop_requested:
                print("Stop requested previously. Exiting application now.")
                QApplication.instance().quit()

        except Exception as e:
            print("\n[ERROR] An exception occurred during simulation:\n")
            traceback.print_exc()
            QMessageBox.critical(self, "Simulation Error", str(e))
        finally:
            self.running = False

# ----------------------
# MAIN
# ----------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaultSimulatorGUI()
    window.show()
    sys.exit(app.exec())
