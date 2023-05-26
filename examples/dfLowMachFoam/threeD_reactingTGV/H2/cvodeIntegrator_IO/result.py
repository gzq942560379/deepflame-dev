#!/usr/bin/env python3

import re
import os
from tabulate import tabulate
from functools import reduce

def get_total_time(time_name:str,log_content:str)->float:
    pattern_str=time_name + "\s*[=:]\s*(\d+\.\d+[eE]?[\+\-]?\d+)"
    time_iter = re.finditer(pattern_str,log_content)
    total_time = 0
    for it in time_iter:
        time = float(it.group(1))
        total_time += time
    return total_time

def get_time_ignore_12(time_name:str,log_content:str)->float:
    pattern_str=time_name + "\s*[=:]\s*(\d+\.\d+[eE]?[\+\-]?\d+)"
    time_iter = re.finditer(pattern_str,log_content)
    total_time = 0
    time_list = []
    for it in time_iter:
        time = float(it.group(1))
        time_list.append(time)
    assert(len(time_list) > 2)
    time_list[0] = 0
    time_list[1] = 0
    total_time = reduce(lambda x, y : x + y, time_list)
    return total_time

def get_iter_info_by_solver_name(solver_name:str,log_content:str):
    pattern_str=r"{}:  Solving for (.*), Initial residual = \d+(\.\d+)?([eE][\+\-]?\d+)?, Final residual = \d+(\.\d+)?([eE]?[\+\-]?\d+)?, No Iterations (\d+)".format(solver_name)
    match_iter = re.finditer(pattern_str,log_content)
    total_tier = 0
    solves = set()
    for it in match_iter:
        iter_count = int(it.group(6))
        solve = it.group(1)
        solves.add(solve)
        total_tier += iter_count
    return total_tier, solves

def get_iter_info_by_solve_name(solve_name:str,log_content:str):
    pattern_str=r"(.*):  Solving for {}, Initial residual = \d+(\.\d+)?([eE][\+\-]?\d+)?, Final residual = \d+(\.\d+)?([eE]?[\+\-]?\d+)?, No Iterations (\d+)".format(solve_name)
    match_iter = re.finditer(pattern_str,log_content)
    total_tier = 0
    solvers = set()
    for it in match_iter:
        iter_count = int(it.group(6))
        solver = it.group(1)
        solvers.add(solver)
        total_tier += iter_count
    return total_tier,solvers


def check_end(log_content:str)->bool:
    if re.search("\nEnd\n", log_content):
        return True
    else:
        return False

def check_run_log(og_content:str)->bool:
    if re.search("topoSet", log_content):
        return False
    if re.search("refineMesh", log_content):
        return False
    if re.search("decomposeFieldsPar", log_content):
        return False
    return True


def parse_log(log_content:str,output_file:str):
    time_total = get_total_time("Total time",log_content)
    time_DNN = get_time_ignore_12("Chemical sources",log_content)
    time_Species_Equations = get_time_ignore_12("Species Equations",log_content)
    time_U_Equations = get_time_ignore_12("U Equations",log_content)
    time_p_Equations = get_time_ignore_12("p Equations",log_content)
    time_Energy_Equations = get_time_ignore_12("Energy Equations",log_content)
    time_thermo_Trans_Properties = get_time_ignore_12("thermo & Trans Properties",log_content)
    time_Diffusion_Correction = get_time_ignore_12("Diffusion Correction Time",log_content)
    time_other = time_total - time_DNN - time_Species_Equations \
        - time_U_Equations - time_p_Equations - time_Energy_Equations - time_thermo_Trans_Properties \
        - time_Diffusion_Correction

    table_header_total = ["" , "time(s)", "percentage(%)"]
    table_data_total = [
        ("Total time", time_total, time_total * 100 / time_total),
        ("DNN", time_DNN, time_DNN * 100 / time_total),
        ("Species_Equations", time_Species_Equations, time_Species_Equations * 100 / time_total),
        ("U Equations", time_U_Equations, time_U_Equations * 100 / time_total),
        ("p Equations", time_p_Equations, time_p_Equations * 100 / time_total),
        ("Energy Equations", time_Energy_Equations, time_Energy_Equations * 100 / time_total),
        ("thermo & Trans Properties", time_thermo_Trans_Properties, time_thermo_Trans_Properties * 100 / time_total),
        ("Diffusion Correction", time_Diffusion_Correction, time_Diffusion_Correction * 100 / time_total),
        ("Other", time_other, time_other * 100 / time_total),
    ]

    time_U_total = get_total_time("U total Time",log_content)
    time_U_build = get_total_time("U build Time",log_content)
    time_U_convert = get_total_time("U convert Time",log_content)
    time_U_solve = get_total_time("U solve Time",log_content)
    time_U_other = get_total_time("U other Time",log_content)
    table_header_UEqn = ["" , "time(s)", "percentage(%)"]
    table_data_UEqn = [
        ("U total Time", time_U_total, time_U_total * 100 / time_U_total),
        ("U build Time", time_U_build, time_U_build * 100 / time_U_total),
        ("U convert Time", time_U_convert, time_U_convert * 100 / time_U_total),
        ("U solve Time", time_U_solve, time_U_solve * 100 / time_U_total),
        ("U other Time", time_U_other, time_U_other * 100 / time_U_total),
    ]

    time_Y_total = get_total_time("Y total Time",log_content)
    time_Y_build = get_total_time("Y build Time",log_content)
    time_Y_convert = get_total_time("Y convert Time",log_content)
    time_Y_solve = get_total_time("Y solve Time",log_content)
    time_Y_other = get_total_time("Y other Time",log_content)
    table_header_YEqn = ["" , "time(s)", "percentage(%)"]
    table_data_YEqn = [
        ("Y total Time", time_Y_total, time_Y_total * 100 / time_Y_total),
        ("Y build Time", time_Y_build, time_Y_build * 100 / time_Y_total),
        ("Y convert Time", time_Y_convert, time_Y_convert * 100 / time_Y_total),
        ("Y solve Time", time_Y_solve, time_Y_solve * 100 / time_Y_total),
        ("Y other Time", time_Y_other, time_Y_other * 100 / time_Y_total),
    ]

    time_E_total = get_total_time("E total Time",log_content)
    time_E_build = get_total_time("E build Time",log_content)
    time_E_convert = get_total_time("E convert Time",log_content)
    time_E_solve = get_total_time("E solve Time",log_content)
    time_E_other = get_total_time("E other Time",log_content)
    table_header_EEqn = ["" , "time(s)", "percentage(%)"]
    table_data_EEqn = [
        ("E total Time", time_E_total, time_E_total * 100 / time_E_total),
        ("E build Time", time_E_build, time_E_build * 100 / time_E_total),
        ("E convert Time", time_E_convert, time_E_convert * 100 / time_E_total),
        ("E solve Time", time_E_solve, time_E_solve * 100 / time_E_total),
        ("E other Time", time_E_other, time_E_other * 100 / time_E_total),
    ]

    time_p_total = get_total_time("p total Time",log_content)
    time_p_build = get_total_time("p build Time",log_content)
    time_p_convert = get_total_time("p convert Time",log_content)
    time_p_solve = get_total_time("p solve Time",log_content)
    time_p_other = get_total_time("p other Time",log_content)
    table_header_pEqn = ["" , "time(s)", "percentage(%)"]
    table_data_pEqn = [
        ("p total Time", time_p_total, time_p_total * 100 / time_p_total),
        ("p build Time", time_p_build, time_p_build * 100 / time_p_total),
        ("p convert Time", time_p_convert, time_p_convert * 100 / time_p_total),
        ("p solve Time", time_p_solve, time_p_solve * 100 / time_p_total),
        ("p other Time", time_p_other, time_p_other * 100 / time_p_total),
    ]

    Ux_iter, Ux_solver = get_iter_info_by_solve_name("Ux",log_content)
    Uy_iter, Uy_solver = get_iter_info_by_solve_name("Uy",log_content)
    Uz_iter, Uz_solver = get_iter_info_by_solve_name("Uz",log_content)

    H_iter, H_solver = get_iter_info_by_solve_name("H",log_content)
    O_iter, O_solver = get_iter_info_by_solve_name("O",log_content)
    H2O_iter, H2O_solver = get_iter_info_by_solve_name("H2O",log_content) 
    OH_iter, OH_solver = get_iter_info_by_solve_name("OH",log_content) 
    O2_iter, O2_solver = get_iter_info_by_solve_name("O2",log_content) 
    H2_iter, H2_solver = get_iter_info_by_solve_name("H2",log_content) 
    ha_iter, ha_solver = get_iter_info_by_solve_name("ha",log_content) 
    p_iter, p_solver = get_iter_info_by_solve_name("p",log_content) 
    solve_total_iter = Ux_iter + Uy_iter + Uz_iter + H_iter + O_iter + H2O_iter + OH_iter \
        + O2_iter + H2_iter + ha_iter + p_iter

    table_header_solve_iter = ["Solve", "Solver" , "iteration"]
    table_data_solve_iter = [
        ("Total", "", solve_total_iter),
        ("Ux",",".join(Ux_solver), Ux_iter),
        ("Uy",",".join(Uy_solver), Uy_iter),
        ("Uy",",".join(Uz_solver), Uz_iter),
        ("H",",".join(H_solver), H_iter),
        ("O",",".join(O_solver), O_iter),
        ("H2O",",".join(H2O_solver), H2O_iter),
        ("OH",",".join(OH_solver), OH_iter),
        ("O2",",".join(O2_solver), O2_iter),
        ("H2",",".join(H2_solver), H2_iter),
        ("H2",",".join(H2_solver), H2_iter),
        ("ha",",".join(ha_solver), ha_iter),
        ("p",",".join(p_solver), p_iter),
    ]

    with open(output_file,"w") as f:
        f.write(tabulate(table_data_total,table_header_total,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")
        f.write(tabulate(table_data_solve_iter,table_header_solve_iter,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")
        f.write(tabulate(table_data_UEqn,table_header_UEqn,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")
        f.write(tabulate(table_data_YEqn,table_header_YEqn,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")
        f.write(tabulate(table_data_EEqn,table_header_EEqn,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")
        f.write(tabulate(table_data_pEqn,table_header_pEqn,"fancy_grid",numalign="right",floatfmt=".2f"))
        f.write("\n")


if __name__ == "__main__":

    search_dir = "/vol0001/hp210260/guozhuoqiang/DeepFlame/deepflame-dev/examples"

    output_file_name_pattern = re.compile(r"stdout.\d+.0")

    result_file_name = "result.out"

    walk_iter = os.walk(search_dir)  
    for path,dir_list,file_list in walk_iter:
        # f.write(path)  
        for file_name in file_list:  
            if output_file_name_pattern.match(file_name):
                output_file_path = os.path.join(path, file_name)
                result_file_path = os.path.join(path, result_file_name)
                if not os.path.exists(result_file_path):
                    log_content = ""
                    with open(output_file_path, 'r') as f:
                        log_content =f.read()
                    if not check_run_log(log_content):
                        continue
                    if check_end(log_content):
                        parse_log(log_content,result_file_path)
                        print("{} complete !".format(result_file_path))