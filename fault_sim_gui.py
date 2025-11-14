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
#
# The original simulation code was kept intact and only minimal changes were made:
#  - write_results_to_file now appends to "Result.txt" instead of overwriting.
#  - Added GUI, stdout redirection to GUI console, run/stop threading behavior.

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
    for g in circuit['gates']:
        t = g['type'].lower()
        ins, y = g['inputs'], g['output']
        if t == 'and' and sa(y,1) in fault_set:
            dom_map.append((sa(y,1), [sa(a,1) for a in ins if sa(a,1) in fault_set]))
            removed.add(sa(y,1))
        elif t == 'nand' and sa(y,0) in fault_set:
            dom_map.append((sa(y,0), [sa(a,1) for a in ins if sa(a,1) in fault_set]))
            removed.add(sa(y,0))
        elif t == 'or' and sa(y,0) in fault_set:
            dom_map.append((sa(y,0), [sa(a,0) for a in ins if sa(a,0) in fault_set]))
            removed.add(sa(y,0))
        elif t == 'nor' and sa(y,1) in fault_set:
            dom_map.append((sa(y,1), [sa(a,0) for a in ins if sa(a,0) in fault_set]))
            removed.add(sa(y,1))
        elif t == 'xor' and sa(y,1) in fault_set:
            dom_map.append((sa(y,1), [sa(a,1) for a in ins if sa(a,1) in fault_set]))
            removed.add(sa(y,1))
        elif t == 'xnor' and sa(y,0) in fault_set:
            dom_map.append((sa(y,0), [sa(a,1) for a in ins if sa(a,1) in fault_set]))
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
    return {'full': full, 'eq_collapsed': eq_collapsed, 'dom_collapsed': dom_collapsed, 'eq_map': eq_map, 'dom_map': dom_map}

# ----------------------
# STEP 6: TIME-EXPANDED SEQUENTIAL FAULT SIM (5-valued)
# ----------------------

def fault_simulation_serial(full_fault_list, circuit, sorted_gates, fault_free_results_binary, test_vectors, verbose_sample=10):
    def parse_fault(fstr):
        net, sa = fstr.split('/SA')
        return net, int(sa)

    VAL0, VAL1, VALX, VALD, VALB = '0','1','X','D','B'

    def combine_db(a, b):
        if a == VALX or b == VALX:
            return VALX
        if a == VALD and b == VALD:
            return VALD
        if a == VALB and b == VALB:
            return VALB
        if (a == VALD and b == VALB) or (a == VALB and b == VALD):
            return VALX
        if a in (VALD, VALB): return a
        if b in (VALD, VALB): return b
        return VALX

    def eval_gate_5val(gtype, in_vals):
        g = gtype.lower()
        has0 = any(v == VAL0 for v in in_vals)
        has1 = any(v == VAL1 for v in in_vals)
        hasX = any(v == VALX for v in in_vals)
        hasD = any(v == VALD for v in in_vals)
        hasB = any(v == VALB for v in in_vals)

        if g == 'and':
            if has0: return VAL0
            if hasD or hasB:
                # combine DB signals if present
                dpart = VALD if hasD else VAL0
                bpart = VALB if hasB else VAL0
                comb = combine_db(dpart, bpart)
                return comb if comb in (VALD, VALB) else VAL1
            if hasX: return VALX
            return VAL1

        if g == 'or':
            if has1: return VAL1
            if hasD or hasB:
                dpart = VALD if hasD else VAL0
                bpart = VALB if hasB else VAL0
                comb = combine_db(dpart, bpart)
                return comb if comb in (VALD, VALB) else VAL0
            if hasX: return VALX
            return VAL0

        if g == 'not':
            v = in_vals[0] if in_vals else VALX
            if v == VAL0: return VAL1
            if v == VAL1: return VAL0
            if v == VALD: return VALB
            if v == VALB: return VALD
            return VALX

        if g == 'nand':
            a = eval_gate_5val('and', in_vals)
            return eval_gate_5val('not', [a])

        if g == 'nor':
            a = eval_gate_5val('or', in_vals)
            return eval_gate_5val('not', [a])

        if g == 'xor':
            if hasX: return VALX
            good_parity = 0
            bad_parity = 0
            for v in in_vals:
                if v == VAL1:
                    good_parity ^= 1; bad_parity ^= 1
                elif v == VAL0:
                    pass
                elif v == VALD:
                    good_parity ^= 1; bad_parity ^= 0
                elif v == VALB:
                    good_parity ^= 0; bad_parity ^= 1
                else:
                    return VALX
            if good_parity == bad_parity:
                return VAL1 if good_parity == 1 else VAL0
            if good_parity == 1 and bad_parity == 0:
                return VALD
            if good_parity == 0 and bad_parity == 1:
                return VALB
            return VALX

        if g == 'xnor':
            x = eval_gate_5val('xor', in_vals)
            return eval_gate_5val('not', [x])

        return VALX

    driver = {g['output']: g for g in circuit['gates']}
    fanout = {}
    for g in circuit['gates']:
        for inp in g['inputs']:
            fanout.setdefault(inp, []).append(g)

    total_faults = len(full_fault_list)
    detected_faults = set()
    fault_detection_info = {}
    T = len(test_vectors)

    print("\nSTAGE 6 (time-expanded): SEQUENTIAL FAULT SIMULATION (5-valued)")
    print(f"Total faults to simulate: {total_faults}")
    print(f"Total time frames       : {T}")

    for idx, fault in enumerate(full_fault_list, start=1):
        net_fault, stuck = parse_fault(fault)
        stuck_symbol = VAL1 if stuck == 1 else VAL0

        detected = False
        first_detect_frame = None
        first_change_level = None

        dff_gates = [g for g in circuit['gates'] if g['type'].lower() == 'dff']
        dff_outs = [g['output'] for g in dff_gates]
        golden_states = {out: 0 for out in dff_outs}
        faulty_states = {out: VAL0 for out in dff_outs}

        for t in range(T):
            vec = test_vectors[t]
            # golden netvals (binary) - use existing precomputed if available
            golden_netvals = {}
            for p in circuit['inputs']:
                golden_netvals[p] = int(vec.get(p, 0))
            for dout in dff_outs:
                golden_netvals[dout] = int(golden_states.get(dout, 0))
            # evaluate golden combinational (binary) to fill internal nets (cheap pass)
            temp_bin = dict(golden_netvals)
            for g in sorted_gates:
                gt = g['type'].lower()
                out = g['output']
                ins = g['inputs']
                if gt == 'dff':
                    clk_names = {'clk','clock','clkin'}
                    dnet = None
                    for n in ins:
                        if n.lower() not in clk_names:
                            dnet = n; break
                    if dnet is None and ins:
                        dnet = ins[-1]
                    if dnet:
                        temp_bin[f"{out}_D_input"] = int(temp_bin.get(dnet, 0))
                    continue
                invals = [int(temp_bin.get(n, 0)) for n in ins]
                try:
                    newb = int(eval_gate(gt, invals))
                except Exception:
                    newb = 0
                temp_bin[out] = newb
            # prepare faulty 5-valued netvals
            faulty_netvals = {}
            for p in circuit['inputs']:
                v = VAL1 if int(vec.get(p,0)) else VAL0
                if p == net_fault: v = stuck_symbol
                faulty_netvals[p] = v
            for dout in dff_outs:
                faulty_netvals[dout] = faulty_states.get(dout, VAL0)
            # evaluate combinational in 5-valued domain
            for g in sorted_gates:
                gtype = g['type'].lower()
                out = g['output']
                ins = g['inputs']
                in_vals_faulty = []
                for n in ins:
                    if n == net_fault:
                        in_vals_faulty.append(stuck_symbol)
                    else:
                        in_vals_faulty.append(faulty_netvals.get(n, VAL0))
                if gtype == 'dff':
                    clk_names = {'clk','clock','clkin'}
                    d_val = VALX
                    for n in ins:
                        if n.lower() not in clk_names:
                            d_val = faulty_netvals.get(n, VAL0) if n != net_fault else stuck_symbol
                            break
                    if d_val == VALX and ins:
                        n = ins[-1]
                        d_val = faulty_netvals.get(n, VAL0) if n != net_fault else stuck_symbol
                    faulty_netvals[f"{out}_D_input"] = d_val
                    continue
                new_val = eval_gate_5val(gtype, in_vals_faulty)
                if out == net_fault:
                    new_val = stuck_symbol
                faulty_netvals[out] = new_val
            # compare primary outputs for detection
            po_diff = False
            for po in circuit['outputs']:
                golden_po = int(temp_bin.get(po, 0))
                faulty_po = faulty_netvals.get(po, VAL0)
                if faulty_po == VALD:
                    if golden_po == 1:
                        po_diff = True
                elif faulty_po == VALB:
                    if golden_po == 0:
                        po_diff = True
                elif faulty_po == VALX:
                    # conservative: do not mark detection on X alone
                    pass
                else:
                    fbin = 1 if faulty_po == VAL1 else 0
                    if fbin != golden_po:
                        po_diff = True
                if po_diff:
                    break
            if po_diff:
                detected = True
                first_detect_frame = t + 1
                # approximate first_change_level by scanning D/DB net positions
                min_level = None
                for netname, v in faulty_netvals.items():
                    if v in (VALD, VALB):
                        if netname.endswith('_D_input'):
                            outnet = netname[:-len('_D_input')]
                            glev = next((g.get('level') for g in sorted_gates if g['output']==outnet), None)
                        else:
                            glev = next((g.get('level') for g in sorted_gates if g['output']==netname), None)
                        if glev is not None:
                            if min_level is None or glev < min_level:
                                min_level = glev
                first_change_level = min_level
                break
            # update golden and faulty DFF states for next frame
            next_golden_states = {}
            for g in dff_gates:
                out = g['output']
                next_golden_states[out] = int(temp_bin.get(f"{out}_D_input", golden_states.get(out, 0)))
            golden_states = next_golden_states
            next_faulty_states = {}
            for out in dff_outs:
                next_faulty_states[out] = faulty_netvals.get(f"{out}_D_input", faulty_states.get(out, VAL0))
            faulty_states = next_faulty_states

        if detected:
            detected_faults.add(fault)
            fault_detection_info[fault] = {'detected': True, 'by_vector': first_detect_frame, 'first_level': first_change_level}
        else:
            fault_detection_info[fault] = {'detected': False, 'by_vector': None, 'first_level': None}

        if idx % 100 == 0 or idx == total_faults:
            print(f"Simulated {idx}/{total_faults} faults... detected so far: {len(detected_faults)}")

    total_detected = len(detected_faults)
    pct = (100.0 * total_detected / total_faults) if total_faults else 0.0
    print("\nSTAGE 6 (time-expanded) SUMMARY")
    print(f"Total faults simulated : {total_faults}")
    print(f"Detected faults        : {total_detected}")
    print(f"Detection percentage   : {pct:.2f}%")
    sample_detected = list(detected_faults)[:verbose_sample]
    sample_undetected = [f for f, info in fault_detection_info.items() if not info['detected']][:verbose_sample]
    if sample_detected:
        print("Sample detected faults (fault -> frame, first_change_level):")
        for f in sample_detected:
            info = fault_detection_info[f]
            print(f"  {f} -> frame #{info['by_vector']}, first_level={info['first_level']}")
    if sample_undetected:
        print("\nSample undetected faults:")
        for f in sample_undetected:
            print(f"  {f}")
    return fault_detection_info

# ----------------------
# STEP 7: WRITE RESULTS
# ----------------------

def write_results_to_file(circuit, faults_info, detection_map, sorted_gates, filename="Result.txt"):
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
        f.write(f"Sample faults after dominance collapsing    : {faults_info['dom_collapsed'][:20]}\n\n")
        detected = [f for f, d in detection_map.items() if d['detected']]
        undetected = [f for f, d in detection_map.items() if not d['detected']]
        f.write("FAULT SIMULATION SUMMARY\n")
        f.write(f"Detected faults   : {len(detected)}\n")
        f.write(f"Undetected faults : {len(undetected)}\n")
        f.write(f"Detection Rate    : {100.0 * len(detected)/len(detection_map):.2f}%\n\n")
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
            full_faults = faults_info['full']

            # Stage 6
            detection_map = fault_simulation_serial(full_faults, circuit, sorted_gates, sim_results, test_vectors)

            # Stage 7 - append to Result.txt
            write_results_to_file(circuit, faults_info, detection_map, sorted_gates, filename="Result.txt")

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
